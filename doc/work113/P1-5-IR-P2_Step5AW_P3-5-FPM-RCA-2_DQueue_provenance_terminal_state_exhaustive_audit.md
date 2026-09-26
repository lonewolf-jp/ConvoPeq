# P3-5-FPM-RCA-2 — D-Queue 2 Entry Provenance / Terminal-State Exhaustive Audit

- Gate: P3-5-FPM-RCA-2
- Mode: read-only source/evidence audit
- Date: 2026-09-24
- Primary verdict: `PARTIALLY-LOCALIZED`
- Gate disposition: `STOP / SOURCE-UNRESOLVED`
- `D-QUEUE-2-PROVEN`: `NO`
- `CONVOPEQ-3-RECONCILED`: `YES`
- Implementation/build/FPM/Dr. Memory: not performed

## 1. Scope / State Freeze

This gate follows RCA-1 and does not enter design or implementation.

The following are frozen:

- production source changes: 0
- test vehicle changes: 0
- CMake changes: 0
- getter/counter additions: 0
- shutdown, epoch, reclaim, router, Recovery semantic changes: 0
- FPM-M0/M1/M2 reruns: 0
- build: not run
- Dr. Memory: not run

The audit uses the current source, the aggregate source authority, the FPM-Run-1 report, and existing evidence artifacts only. No runtime value is inferred from a field that is absent from the evidence schema.

## 2. `ConvoPeq(3).md` Reconciliation

The user clarified that `ConvoPeq(3).md` means `C:\VSC_Project\ConvoPeq\ConvoPeq.md`. That alias is used as the sole aggregate authority for this gate. The RCA-1 path-availability note is superseded by this clarification and is not carried forward.

Observed authority:

- Path: `ConvoPeq.md`
- Size: 5,535,334 bytes
- Lines: 121,294
- SHA-256: `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609`

The generated source sections for the requested files were compared with the live source after removing only the aggregate Markdown fence/header and normalizing blank/comment lines. The audited non-comment code bodies matched:

| Required symbol/path | Aggregate authority | Live source | Result |
|---|---:|---:|---|
| `AudioEngine::isFullyDrained()` | `ConvoPeq.md:21070` | `src/audioengine/AudioEngine.Threading.cpp:153-213` | PASS |
| `RuntimeIntentCoordinator::ShutdownScheduler::isFullyDrained()` | `ConvoPeq.md:32004` | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:511-562` | PASS |
| `ISRRetireRouter::pendingRetireCount()` | `ConvoPeq.md:30045` | `src/audioengine/ISRRetireRouter.cpp:586-591` | PASS |
| `DeferredDeletionQueue::enqueue()` | generated source section | `src/DeferredDeletionQueue.h:66-107` | PASS |
| `SnapshotCoordinator::finalizeShutdown()` | generated source section | `src/core/SnapshotCoordinator.h:62-72` | PASS |
| `SnapshotCoordinator::retireCurrentAndTarget()` | generated source section | `src/core/SnapshotCoordinator.h:165-184` | PASS |
| `AudioEngine::releaseResources()` | `ConvoPeq.md:17536` | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:34-754` | PASS |
| `waitForDrain(2000, 2)` | `ConvoPeq.md:18117` | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:615-617` | PASS |
| `markTimedOut()` | `ConvoPeq.md:18134` | `src/audioengine/ISRShutdown.cpp:72-112` | PASS |
| `markShutdownComplete()` | `ConvoPeq.md:18226` | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:724`; bridge implementation `:568-579` | PASS |
| `transitionTo(ShutdownComplete)` | generated source section | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:744-745`; FSM `:124-152` | PASS |

Reconciliation result: `CONVOPEQ-3-RECONCILED`.

The aggregate line number is generally two lines above the live line because the generated section has a Markdown header/fence. The code bodies and symbol content match.

## 3. RCA-1 Findings Inherited

RCA-1 established the following, and this gate rechecked each item against the current authority/live source:

- M0, M1, and M2 all completed with process exit code 0.
- All three terminal captures reported `phase=10`, `completed=1`, `blockingReason=9 (Unknown)`, `transitionViolations=0`, `fullyDrained=0`, and `t3b=0`.
- All three terminal drain audits reported `routerPending=2`; the other reported pending fields were zero.
- `evidence/shutdown_trace.json` independently reports `phase=10`, `blockingReason=Unknown`, `sh3_pendingRetire=2`, and `verified=false`.
- `routerPendingRetire` is calculated as `m_retireRouter->pendingRetireCount() + fallbackQueueDepth_` (`AudioEngine.Threading.cpp:120-121`).
- `fallbackQueueDepth_` has one production writer, which writes literal zero (`AudioEngine.h:5064`, `AudioEngine.Retire.cpp:136-140`). Thus the observed `routerPending=2` is the engine `DeferredDeletionQueue` depth, not a fallback contribution.
- `waitForDrain(2000, 2)` returns false; the timeout reason defaults to `Unknown` unless a stuck reader or active builder is observed (`ReleaseResources.cpp:615-633`).
- `finalizeShutdown(true)` retires the SnapshotCoordinator current/target slots and skips its provider `tryReclaim()` (`SnapshotCoordinator.h:62-72`).
- The outer ShutdownRuntime can still transition to `ShutdownComplete` after a timeout; that transition is not an outer `isFullyDrained()` success gate (`ISRShutdown.cpp:124-152`, `ReleaseResources.cpp:724-745`).
- RCA-1 correctly kept the exact identity of the two D-queue entries unresolved.

RCA-2 adds a complete writer inventory and separates the engine D queue from adjacent queues and shutdown fallback stores. It does not convert the aggregate count into per-entry identity.

## 4. DeferredDeletionQueue Ownership / Provider Topology

### 4.1 Engine queue topology

`AudioEngine` owns one `EpochDomain m_epochDomain` and one `ISRRetireRouter` constructed with that provider (`AudioEngine.h:5017-5029`, `AudioEngine.CtorDtor.cpp:21-42`). `SnapshotCoordinator m_coordinator` is also constructed with the same `m_epochDomain` (`AudioEngine.CtorDtor.cpp:26-27`).

`EpochDomain` owns the actual `DeferredDeletionQueue deferredDeletionQueue` (`EpochDomain.h:585-587`). The router is a policy/delegation layer:

```text
AudioEngine m_epochDomain
        │
        ├── EpochDomain::DeferredDeletionQueue (engine D queue)
        │       ├── enqueue / enqueueTyped
        │       ├── reclaim(minReaderEpoch)
        │       └── sizeApprox()
        │
        ├── ISRRetireRouter m_retireRouter
        │       ├── enqueueRetire / enqueueWithRetry
        │       ├── Q / EmergencyQ / TerminalReclaim
        │       └── pendingRetireCount() -> provider D sizeApprox()
        │
        └── SnapshotCoordinator m_coordinator
                └── direct provider enqueue for GlobalSnapshot retirement
```

The D queue stores pointer, deleter, epoch, type, publication sequence, and generation metadata (`DeferredDeletionQueue.h:75-99`). Reclaim is FIFO-strict and deletes an entry only when `entry.epoch < minReaderEpoch` (`DeferredDeletionQueue.h:110-175`).

With no active readers, `getMinReaderEpoch()` starts at `currentEpoch()` and does not lower it (`EpochDomain.h:214-247`). Therefore an entry enqueued at the current epoch is not reclaimable until a later epoch advance. This is a source-proven mechanism for residual D entries, but it does not identify the two observed entries.

### 4.2 Queue separation

The following are not the engine router’s D queue and must not be counted as `routerPending=2`:

- `EQProcessor` owns a separate `EpochDomain m_epochDomain` (`EQProcessor.h:487-497`). `EQProcessor::enqueueDeferredDeleteWithFallback()` creates a stack `ISRRetireRouter` over that local domain (`EQProcessor.Core.cpp:43-61`). Its D queue is not the `AudioEngine::m_epochDomain` queue.
- `SnapshotRetireManager` owns a separate `DeletionQueue` (`SnapshotRetireManager.h:18-50`). Current source search found no production use; it is a legacy/parallel queue, not the engine D queue.
- `LifetimeState`/`RetireRuntimeEx` intent queues are System 1 slot-state paths. They are drained by `drainPendingRetireIntentsForShutdown()` and are explicitly separate from the DeferredDeletionQueue (`ReleaseResources.cpp:798-803`).
- Router Q, EmergencyQ, and TerminalReclaim are post-D fallback authorities. They are not included in `pendingRetireCount()`/D depth unless an entry is still in D.

## 5. D-Queue Enqueue Site Inventory

The table lists production-reachable or source-defined engine-D writer sites. “M0 possible” means source control flow permits the path during the M0 startup/shutdown sequence; it does not claim that the FPM evidence observed that path.

| enqueue site | object / entry type | epoch | caller | shutdown phase | M0 possible | M1/M2 added |
|---|---|---|---|---|---|---|
| `SnapshotCoordinator::retireCurrentAndTarget` current (`SnapshotCoordinator.h:173-177`) | `GlobalSnapshot*`, `Generic` | one `publishEpoch()` result | `finalizeShutdown()` or `SnapshotCoordinator` destructor | S10, after timeout decision | Yes if current is non-null | Yes |
| `SnapshotCoordinator::retireCurrentAndTarget` target (`SnapshotCoordinator.h:179-183`) | `GlobalSnapshot*`, `Generic` | same `retireEpoch` as current | same | S10 | Yes if target is non-null | Yes |
| `SnapshotCoordinator::switchImmediate` old target (`SnapshotCoordinator.h:90-99`) | `GlobalSnapshot*`, `Generic` | `currentEpoch()` | publish/switch path | runtime or shutdown preparation | Source-capable | Yes |
| `SnapshotCoordinator::switchImmediate` old current (`SnapshotCoordinator.h:103-109`) | `GlobalSnapshot*`, `Generic` | `publishEpoch()` | publish/switch path | runtime or shutdown preparation | Source-capable | Yes |
| `SnapshotCoordinator::startFade` old target (`SnapshotCoordinator.cpp:53-61`) | `GlobalSnapshot*`, `Generic` | `currentEpoch()` | fade start | runtime crossfade path | Source-capable, not exercised by FPM M0 sequence | M2 source-capable |
| `SnapshotCoordinator::completeFade` old current (`SnapshotCoordinator.cpp:100-119`) | `GlobalSnapshot*`, `Generic` | `publishEpoch()` | fade completion | runtime crossfade path | Source-capable, not exercised by FPM M0 sequence | M2 source-capable |
| `SnapshotCoordinator::resetFadeStateAndRetireTarget` (`SnapshotCoordinator.cpp:81-95`) | `GlobalSnapshot*`, `Generic` | `publishEpoch()` | fade/reset path | runtime or shutdown preparation | Source-capable | Source-capable |
| `DSPLifetimeManager::retire` (`DSPLifetimeManager.cpp:40-72`) | registered `AudioEngine::DSPCore*`, `Generic` | publication epoch if supplied, otherwise router current epoch | publish completion, crossfade transition, rollback/final DSP paths | runtime and terminal disposition | Yes | Yes |
| `DSPLifetimeManager::retireByHandle` (`DSPLifetimeManager.cpp:79-136`) | `DSPCore*`, `Generic` | router `currentEpoch()` | Observe, Recovery, shutdown handle disposition | Coordinator/terminal paths | Source-capable | Recovery/observe paths added in M1/M2 |
| `AudioEngine::enqueueDeferredDeleteNonRtWithResult` (`AudioEngine.h:4460-4496`) | `RuntimePublishWorld`, `CacheMap`, `IRState`, `StereoConvolver`, or other caller pointer; `World` for published-world path, otherwise caller type | `markRetireEpoch()` unless caller path is already shutdown-sensitive | cache replacement, published/rejected world retirement, convolver lifecycle | During shutdown the wrapper routes through `shutdownReclaim`/Terminal, not normal D | Source-capable | Source-capable |
| `ISRRetireRouter::retireRT` (`ISRRetireRouter.cpp:281-288`) | caller pointer, normally `Generic` | provider `currentEpoch()` | RT-safe router interface / legacy templates | Any RT producer before shutdown | Source-capable, no FPM attribution | Source-capable |
| `RuntimeIntentCoordinator::enqueueRetire` (`ISRRuntimePublicationCoordinator.cpp:148-172`) | caller pointer, `Generic` | caller-supplied | caller supplies an `ISRRetireRouter` | depends on caller | No engine-D call site proven; current production call found uses EQProcessor’s separate stack router | Same limitation |
| `EpochDomain::enqueueRetire/enqueueRetireTyped` (`EpochDomain.h:399-410`) | final D entry | caller-supplied | router/provider terminal | any | Sink, not an independent provenance site | Sink |

Indirect production callers of `DSPLifetimeManager::retire` include `DSPTransition::onPublishCompleted` (`DSPTransition.h:49-154`), `RuntimePublicationOrchestrator` build-failure/registered-DSP paths (`RuntimePublicationOrchestrator.cpp:177-193`, `:245-262`, `:745-762`), `AudioEngine.Timer` retire paths (`Timer.cpp:1937-1995`), and shutdown final DSP paths (`ReleaseResources.cpp:568-588`). These are source-backed candidate families, not per-entry observations.

The current authority verifier also identifies the two shutdown final-DSP calls at `ReleaseResources.cpp:577` and `:586` as retire call sites. Its warning is an inventory warning, not evidence that either call produced a residual D entry.

### 5.1 Shutdown-sensitive writer paths that do not necessarily enter D

`AudioEngine::enqueueDeferredDeleteNonRtWithResult()` first checks `isShutdownInProgress()` (`AudioEngine.h:4466-4475`). During shutdown it transfers ownership to `m_retireRouter->shutdownReclaim()`, which goes to TerminalReclaim and may synchronously destroy when epoch-safe (`ISRRetireRouter.cpp:570-584`). This applies to the published-world clear and owner-channel residual callback in the terminal path.

The owner-channel callback is especially important because it runs after `m_coordinator.finalizeShutdown(timedOut)` (`ReleaseResources.cpp:664`, `:677-693`). It is a post-finalize writer site, but under the shutdown flag it is a TerminalReclaim candidate rather than an engine-D entry. Its runtime count is not present in the FPM schema.

## 6. Exact Provenance of the 2 Entries

### 6.1 What source proves

`finalizeShutdown(true)` executes `retireCurrentAndTarget()` before the timed-out reclaim skip (`SnapshotCoordinator.h:62-72`). That helper:

1. calls `publishEpoch()` once;
2. exchanges current to null and enqueues it if non-null;
3. exchanges target to null and enqueues it if non-null;
4. uses the same captured `retireEpoch` for both entries;
5. sends failed D enqueue attempts to the router quarantine sink.

Therefore source proves that `finalizeShutdown(true)` can enqueue zero, one, or two `Generic` `GlobalSnapshot` entries into the shared D queue, depending on the two slot values.

### 6.2 What source does not prove

`SnapshotSlotStore` initializes both slots to null and exposes independent atomic current/target pointers (`SnapshotSlotStore.h:27-33`, `:35-53`, `:55-57`). The finalize helper has null checks but no runtime occupancy capture and no per-entry identity output.

Other engine-D writer families can run before or during shutdown:

- startup/publication snapshot replacement;
- publish completion and crossfade DSP retirement;
- cache and convolver object retirement;
- final active/fading DSP retirement;
- Observe/Recovery handle retirement;
- direct `SnapshotCoordinator` fade/reset paths.

The FPM schema reports only aggregate `routerPending=2`. It does not report:

- D entry pointer/object identity;
- `DeletionEntryType`;
- enqueue epoch;
- enqueue site;
- current/target slot occupancy at S10;
- number of entries added by owner-channel/world-clear paths;
- the loop-time value of any Layer-1 predicate before timeout.

`evidence/retire_trace_shutdown_last.json` reports five reclaimed slot lifecycles and no D-entry identity. It cannot be used to reconstruct the two D entries.

### 6.3 Provenance verdict

```text
D-QUEUE-2-PROVEN = NO
```

The current/target pair is a source-compatible explanation for the terminal count, but it is not the only source-compatible explanation. The exact two-entry provenance remains unobservable with the existing API/evidence and without a prohibited getter/counter or vehicle change.

## 7. `finalizeShutdown(true)` Before/After State

The requested S0-S13 timeline is below. “D depth” means engine `DeferredDeletionQueue::sizeApprox()` when directly known; “not captured” means no existing artifact records the value at that point.

| point | source action | D-queue / epoch / reader state | reclaim and enqueue conclusion |
|---|---|---|---|
| S0 graceful drain start | `ReleaseResources.cpp:279-311`; close registration, escalate intents, poll up to 5 s | Entry depth not captured; loop tests router D count and active readers | Per cycle: OverflowRing reinject, `publishEpoch()`, `tryReclaim()` |
| S1 ForceEpochAdvance | `ReleaseResources.cpp:262-265`; `advanceRetireEpoch()` | `EpochDomain::publishEpoch()` increments global epoch; no separate target epoch | D entries are not automatically reclaimed by the advance itself |
| S2 ReclaimComplete | `ReleaseResources.cpp:373-376` | D depth after drain not captured | `drainDeferredRetireQueues(true)` performs provider reclaim, Q/E/T drain, and pending-handle retry |
| S3 EmergencyDrain | `ReleaseResources.cpp:378-429` | D depth not captured | Phase always entered; body is conditional. Default path is diagnostic-only, with no unconditional D enqueue |
| S4 VerifyDrained | `ReleaseResources.cpp:474-476` | Audit fields collected, but no state snapshot is persisted for this gate | This is an audit point, not an outer success gate |
| S5 terminal disposition | `ReleaseResources.cpp:478-588`; active/fading handles, world clear, final DSP lifetime retire | D depth before/after each operation not captured | Direct `DSPLifetimeManager::retire` can enter D; shutdown-sensitive wrapper paths can go to Terminal |
| S6 wait start | `ReleaseResources.cpp:615`; `waitForDrain(2000,2)` | Loop-time D depth and all predicate values not captured | Each iteration calls `drainDeferredRetireQueues(true)` while `isFullyDrained()` is false |
| S7 wait end | `Threading.cpp:231-246`; timeout return | Timeout is certain from FPM result; exact last-iteration predicate vector is not captured | `timedOut=true` is derived from `drainedWithinBudget=false` |
| S8 System 1 drain | `ReleaseResources.cpp:618-622`, implementation `:803-887` | D depth is not the System 1 queue; D value not captured | Drains OverflowRing, LifetimeState MPSC/fallback, and bounded re-injection; not a direct D-entry identity source |
| S9 timeout decision | `ReleaseResources.cpp:624-633`; `ISRShutdown.cpp:72-112` | `Unknown` is the default; FPM has no stuck reader and no active builder | `markTimedOut(Unknown)` stores reason and changes phase to `TimedOut` |
| S10 finalize | `ReleaseResources.cpp:654-664`; `SnapshotCoordinator.h:62-72` | `timedOut=true`: current/target are retired, then provider `tryReclaim()` is skipped | This is the strongest late-enqueue mechanism, but current/target occupancy is not captured |
| S10b post-finalize owner residual | `ReleaseResources.cpp:677-697` | `drainedResidual` count not in FPM | Calls shutdown-sensitive world retirement; normally TerminalReclaim, not D |
| S11 bridge completion | `ReleaseResources.cpp:724`; `ISRRuntimePublicationCoordinator.cpp:568-579` | Bridge checks its own 14-predicate `ShutdownScheduler::isFullyDrained()` | Bridge may become `Faulted`; this is not the outer Layer-1 `AudioEngine::isFullyDrained()` gate |
| S12 phase transition | `ReleaseResources.cpp:744-745`; `ISRShutdown.cpp:124-152` | Phase becomes `ShutdownComplete`; D value not checked by transition | Transition may skip terminal phases TimedOut/Failed and records no drain success |
| S13 FPM capture | `P1PolyphaseGainCharacterization.cpp:1317-1351` | `routerPending=2`, `quarRes=0`, active readers 0, `fullyDrained=0` | Capture is post-release and post-finalize; it is not a loop-time or per-entry observation |

The current epoch is `EpochDomain::globalEpoch` (`EpochDomain.h:186-201`). There is no independent target-epoch counter in this shutdown path. The current/target terms in `SnapshotCoordinator` are pointer slots, not separate epoch domains.

## 8. `pendingReclaimHandles_` Audit

### 8.1 Lifecycle

| operation | source site | result |
|---|---|---|
| declaration | `AudioEngine.h:5118-5130` | default-empty `std::vector<ReclaimIdentity>` plus mutex |
| initial value | member declaration | default-empty; no constructor write |
| insert: requestReclaim false after TOCTOU | `AudioEngine.h:4607-4615` | `ReclaimIdentity{handle, retireEpoch}` pushed under mutex |
| insert: epoch unsafe | `AudioEngine.h:4618-4624` | same identity form pushed under mutex |
| insert: retry false | `AudioEngine.Retire.cpp:115-123` | pushed under mutex |
| insert: retry still unsafe | `AudioEngine.Retire.cpp:125-131` | pushed under mutex |
| swap-out | `AudioEngine.Retire.cpp:90-98` | `pending.swap(pendingReclaimHandles_)` empties the member before retry |
| clear/erase | source search | no direct `.clear()` or `.erase()` on this member |
| shutdown drain | `AudioEngine.Threading.cpp:235-244`, `ReleaseResources.cpp:654-662` | `drainDeferredRetireQueues(true)` retries the swapped identities; it does not unconditionally discard failed identities |
| isFullyDrained read | `AudioEngine.Threading.cpp:198-205` | mutex-protected `.empty()` is a required conjunct |

### 8.2 Classification

```text
pendingReclaimHandles_ = SOURCE-UNRESOLVED
```

The source proves the predicate read and the retry lifecycle. It does not prove that the vector was empty at the FPM terminal capture: `fullyDrained=0`, the loop timed out, and the FPM schema has no `pendingReclaimHandles_.size()` field. A successful `isFullyDrained()==true` would require the vector to be empty at that read, but this run did not establish that condition.

`PROVEN-EMPTY` is therefore rejected for the observed terminal state. `PROVEN-NONEMPTY` is also not supported.

## 9. Quarantine / TerminalReclaim Audit

These are three different domains:

| name | source expression | physical meaning | FPM evidence |
|---|---|---|---|
| `quarRes` / `dspQuarantineManager_.residentCount()` | `AudioEngine.Threading.cpp:126`; `ReleaseResources.cpp:447-470` | DSP quarantine manager’s active quarantine slots | FPM `quarRes=0` |
| `retireQuarantineResident` | `AudioEngine.Threading.cpp:181-182`; `ISRRetireRouter.cpp:431-435` | Router Q + EmergencyQ resident count | Not emitted by FPM |
| `terminalReclaimResident` | `AudioEngine.Threading.cpp:190-191`; `ISRRetireRouter.cpp:535-539` | TerminalReclaimAuthority resident count | Not emitted by FPM |

Shutdown paths are separate:

- Q and EmergencyQ are drained by `drainQuarantineStore()` / `drainEmergencyAndTerminal()` (`ISRRetireRouter.cpp:386-394`, `:550-562`).
- Terminal is drained by `drainTerminalReclaim()` and force-drained by `drainAllQuarantineStore()` / `drainAll()` (`ISRRetireRouter.cpp:443-451`, `:541-562`, `:593-601`).
- DSP quarantine slots are destroyed through `dspQuarantineManager_.destroyForShutdown()` (`ReleaseResources.cpp:447-461`).

Therefore:

```text
FPM quarRes = 0
retireQuarantineResident = UNOBSERVED
terminalReclaimResident = UNOBSERVED
```

`quarRes=0` must not be interpreted as proof that Q/E/T or TerminalReclaim was empty.

## 10. RuntimePublicationBridge Exhaustive State Audit

`runtimePublicationBridge_` is a `RuntimeIntentCoordinator` (`AudioEngine.h:5182-5185`). Its `ShutdownScheduler::isFullyDrained()` is Layer 2 and has 14 conjuncts (`ISRRuntimePublicationCoordinator.cpp:511-562`). It is distinct from Layer 1’s `AudioEngine::isFullyDrained()`.

The table uses:

- `source-capable`: a writer exists in the M0 source path, but no M0 per-field evidence exists;
- `path-only`: a shutdown clear/discard path exists, but the terminal value is not emitted;
- `no`: the existing FPM/evidence schema does not determine the value.

| bridge predicate | M0 generation / source path | shutdown clear or close path | terminal value source-only確定 |
|---|---|---|---|
| `swapPending_` | `commit()` sets true then false (`Coordinator.cpp:109-120`); source-capable during startup/publish | No unconditional shutdown reset; normal commit close is not a terminal observation | no |
| `intentQueue_` | Observe/Quarantine/Recovery intent producers; source-capable | `processIntent()` pops entries; no unconditional post-join clear of every common-queue entry | no |
| `observeDeferredRing_` | Observe overflow producer; source-capable | `drainObserveDeferred()` consumes entries; no post-join terminal snapshot | no |
| `quarantineFallbackQueue_` | `submitQuarantine()` fallback producer; source-capable | `processIntent()` drains fallback first; no separate final clear artifact | no |
| `recoveryIntentQueue_` | M0 has no FPM Recovery episode; M1/M2 path-capable | `discardRecoveryRequestsOnShutdown()` explicitly pops/discards after Builder join (`Coordinator.cpp:1457-1471`) | no; clear path only |
| `retireBacklogCount_` | `onRetireAccepted/onRetireConsumed`; source-capable | event decrement only; no shutdown reset | no |
| `publicationBacklogCount_` | publish lifecycle writer; source-capable | event decrement only; no shutdown reset | no |
| `publicationIntentResidencyCount_` | Publish intent reservation; source-capable during startup/publish | `processIntent()` decrements on Publish pop | no |
| `pendingIntentCount_` | Observe/Quarantine/Recovery reservation; source-capable | pop/discard paths decrement; no FPM field | no |
| `reclaimInFlightCount_` | `onReclaimBegin/End()` around drains; source-capable | paired drain operations; no FPM field | no |
| `quarantineIntentResidencyCount_` | `submitQuarantine()` primary reservation; source-capable | `processIntent()` decrements on Quarantine pop | no |
| `quarantineRingResidencyCount_` | fallback reservation; source-capable | `processIntent()` decrements on fallback pop | no |
| `recoveryAdmissionPending_` | M0 no Recovery episode; M1/M2 path-capable | `discardPendingRecoveryAdmission()` explicitly writes false after join (`Coordinator.cpp:1316-1325`) | no; value not emitted |
| `liveLogicalRecoveryObligationCount()` | M0 no Recovery episode; M1/M2 path-capable | `discardRecoveryRequestsOnShutdown()` resolves Live obligations as ShutdownDiscarded | no; value not emitted |

No bridge predicate may be assigned a terminal numeric value from the current FPM schema. The outer `isFullyDrained()` failure could have occurred in Layer 1, Layer 2, or an earlier short-circuited predicate; the aggregate `routerPending=2` does not identify the loop-time blocker.

## 11. M0/M1/M2 Differential Interpretation

| run | source actions | published / retired | routerPending | interpretation |
|---|---|---:|---:|---|
| M0 | startup/settle, stop, terminal capture; no FPM Recovery, explicit publish, or crossfade operation | 3 / 2 | 2 | Constant residue already exists without Recovery/Publish/Crossfade episode |
| M1 | M0 plus one Recovery episode, then stop | 4 / 3 | 2 | Recovery is not necessary for the constant residue; episode counters were clean |
| M2 | M0 plus publish-1, crossfade settle, Recovery, publish-2, then stop | 6 / 5 | 2 | Additional publish/crossfade activity does not change terminal router depth |

Source interpretation:

- Recovery is ruled out as a necessary cause of the constant `routerPending=2`.
- Crossfade is ruled out as a necessary cause: M0 has no explicit crossfade action and M2 settles to `xfade=0`.
- Per-operation publish accumulation is not supported as the constant explanation: published/retired counts change while router depth remains 2, and graceful drain advances/reclaims epochs.
- Publish/snapshot retirement remains a possible contributor to the exact residual entries, especially because startup publication exists even in M0. It is not proven to be the source of the two entries.
- The evidence does not distinguish “two late SnapshotCoordinator entries” from “one late entry plus one earlier DSP/pointer entry” or another shared-D writer combination.

## 12. T3a / T3b Terminal Contract Re-validation

The vehicle implements the contract as follows (`P1PolyphaseGainCharacterization.cpp:1331-1335`):

```text
T3a = phase == ShutdownComplete
   && completed
T3b = T3a
   && blockingReason == None
   && transitionViolations == 0
   && AudioEngine::isFullyDrained()
```

Observed state for M0/M1/M2:

| condition | value | result |
|---|---:|---|
| `phase == ShutdownComplete` | 1 | T3a true |
| `completed` | 1 | true |
| `blockingReason == None` | 0 (`Unknown`) | false |
| `transitionViolations == 0` | 1 | true |
| `isFullyDrained()` | 0 | false |
| `t3b` | 0 | `INVALID-TERMINAL` |

State transition:

```text
waitForDrain(2000, 2) timeout
        ↓
timedOut = true
        ↓
markTimedOut(Unknown)
        ↓
phase = TimedOut
        ↓
finalizeShutdown(true)
        ↓
markShutdownComplete()       [bridge-local check only]
        ↓
transitionTo(ShutdownComplete)
        ↓
FPM capture: T3a=true, T3b=false
```

`transitionTo()` permits the transition because the skipped intermediate phases are terminal phases; it does not call outer `isFullyDrained()` (`ISRShutdown.cpp:124-152`). `markShutdownComplete()` checks only the bridge’s own 14-predicate scheduler and can set the bridge state to `Faulted`; it is not an outer Layer-1 success gate (`ISRRuntimePublicationCoordinator.cpp:568-579`). Thus `ShutdownComplete` is not equivalent to a successful drain.

## 13. Root-Cause Localization Verdict

### Reconciliation

```text
CONVOPEQ-3-RECONCILED = YES
```

### D-queue

```text
routerPending = 2
fallbackQueueDepth = 0
engine DeferredDeletionQueue depth = 2
D-QUEUE-2-PROVEN = NO
```

The count and shared-queue topology are source/evidence-backed. The two entries’ exact object/type/epoch/caller identity is not source-observable from existing artifacts.

### Reclaim handle

```text
pendingReclaimHandles_ = SOURCE-UNRESOLVED
```

### Quarantine domains

```text
FPM quarRes = 0
retireQuarantineResident = UNOBSERVED
terminalReclaimResident = UNOBSERVED
```

### Primary verdict

```text
PARTIALLY-LOCALIZED
```

This is the ceiling permitted by the read-only constraints. `LOCALIZED` is not supportable because exact entry provenance and simultaneous loop-time values of the other Layer-1/Layer-2 predicates are absent. `UNLOCALIZED` is not the appropriate primary classification because the D-queue depth mapping, timeout path, late-finalize mechanism, reclaim suppression, and terminal transition are source-backed.

## 14. Remaining Unresolved Items

1. Exact identity, type, epoch, and enqueue site of the two D-queue entries.
2. The D-queue depth and predicate vector at the instant `waitForDrain` timed out, as distinct from the post-finalize capture.
3. `pendingReclaimHandles_.size()` and per-identity state at S6/S7/S13.
4. Terminal values of all 14 RuntimePublicationBridge predicates, especially the common intent/observe/quarantine queues and semantic counters.
5. Whether `drainedResidual`/owner-channel activity contributed any terminal authority residency; the current shutdown-sensitive path normally routes it to TerminalReclaim, but its count is not in FPM evidence.
6. The original temporary FPM stdout files are not present as standalone files in the current workspace; Step5AU preserves their quoted output, and `shutdown_trace.json` independently corroborates the key terminal values.
7. The current source graph contains multiple legitimate engine-D writer families. Selecting one by plausibility would be an inference, not evidence.

## 15. STOP / Next Gate

STOP is triggered by the exact-provenance condition:

```text
D-queue 2 entry provenance cannot be determined with existing source/evidence APIs.
```

No getter, counter, telemetry extension, shutdown semantic change, epoch/reclaim/router/Recovery change, build, FPM rerun, or Dr. Memory run was performed.

The next work item must remain a separately authorized evidence/observability gate. This RCA-2 artifact does not authorize an implementation design or an FPM rerun.

## Appendix A. Evidence and Tool Ledger

- Source authority: `ConvoPeq.md`, 121,294 lines, SHA-256 recorded in §2.
- Prior measurement: `doc/work113/P1-5-IR-P2_Step5AU_P3-5-FPM-Run-1_full_pipeline_measurement.md`.
- Prior RCA: `doc/work113/P1-5-IR-P2_Step5AV_P3-5-FPM-RCA-1_terminal_drain_failure_root_cause_localization.md`.
- Runtime evidence: `evidence/shutdown_trace.json`, `evidence/retire_trace_shutdown_last.json`, `evidence/retire_timeline.json`, `evidence/isr_v73_shutdown_reclaim_report.json`.
- Code search: Semble, AiDex session/signatures, Cocoindex status/search, `rg`, ast-grep, `fdfind`, `sed`, `awk`, and `ag` where available.
- Authority scan: `tools/retire_authority_verifier.py`; it reported the two shutdown final-DSP call sites at `ReleaseResources.cpp:577` and `:586`.
- Static checks: targeted cppcheck completed without a reported diagnostic in the sampled files. clang-tidy was not used as evidence because the WSL-to-Windows compile-database path was incompatible; no build was run.
- Index freshness: Cocoindex status was available; tgrep was stale (`.tgtep`, two days old); no Graphify graph existed, so no Graphify query result was used.
- Prohibited actions: no FPM-M0/M1/M2 execution, no build, no Dr. Memory, and no source/vehicle/CMake edits.

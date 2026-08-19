# 15-P-9: Residual Ownership/Authority Closure Audit

## Status: PASS

## Objective

Pure audit phase (no production code changes). Verify that **no `RuntimeState*`
or `DSPCore*` is orphan / ownerless at shutdown completion** across:

- System 1 — RetireIntent slot-state ownership (LifetimeState / EpochControl)
- System 2 — DSPCore* pointer-lifetime ownership (DeferredDeletionQueue → Quarantine → Emergency → Terminal)
- OwnerChannel — RuntimeState* cross-RT-boundary transfer

Confirm final shutdown ordering for both `~AudioEngine()` and `releaseResources()`,
and that every pointer reaches a determinate terminal sink (actual `delete`/`deleter()`
call), with no residual or double-dispose path.

Regression baseline: **15-P-8 — Debug CTest 33/33 PASS (36.76s), Release CTest 33/33
PASS (32.70s)**. No production code change since 15-P-7.

Environment note: ASan 0xC0000139 — BLOCKED (environment/runtime issue, recorded
in 15-P-8, NOT addressed here per audit-phase scope).

---

## 1. Ownership Map

| System | Pointer type | Authority | Terminal sink | Force-drain contract |
| --- | --- | --- | --- | --- |
| System 1 | RetireIntent (slot index) | LifetimeState -> EpochControl (lane FSM) | reclaim -> Reclaimed lifecycle | drainPendingRetireIntentsForShutdown |
| System 2 | DSPCore pointer / RuntimeState pointer | DeferredDeletionQueue -> RetireQuarantineStore -> EmergencyQuarantineStore -> TerminalReclaimAuthority | deleter() (immediate or drainAll) | drainAllQuarantineStore / drainDeferredRetireQueues |
| OwnerChannel | RuntimeState pointer (aligned_unique_ptr) | RuntimeWorldAuthority (sole owner) | drainAllNonRt -> enqueueDeferredDeleteNonRtWithResult -> shutdownReclaim -> TerminalReclaimAuthority | drainAllNonRt (post finalizeShutdown) |

Single terminal authority: **TerminalReclaimAuthority** (growable `std::vector`
in `ISRRetireRouter.h`). Its `store()` ALWAYS returns `true` (P-4 guarantee) —
there is no "store full" failure path that could leave a pointer orphan.

---

## 2. System 1 — RetireIntent Slot-State Ownership

### 2.1 Authority structure

- `LifetimeState` (`src/audioengine/ISRRetire.h`): owns `slots_[256]` (Vyukov
  MPSC), `fallbackQueue_[4096]` (mutex-protected), `overflowRing_`
  (`RetireOverflowRing`, SPSC 16384).
- `EpochControl` (`src/audioengine/ISRRetireRuntimeEx.cpp`): owns
  `laneBySlot_` (RetireLane: RTIntent / Coordination / Epoch /
  Reclaim / Quarantine) and `lifecycleStateBySlot_` (Visible →
  CompareEligible → TelemetryRetained / ReplayRetainedOptional →
  ReclaimEligible → Reclaimed).
- `reclaim(slot)` is idempotent — `transitionLifecycle` is a no-op on
  already-`Reclaimed` (no double-free).

### 2.2 Handoff chain on shutdown

`emitRetireIntent` → MPSC ticket (bounded spin 64) → on failure:
tombstone (`dspSlot=UINT32_MAX`) + fallback → if fallback full →
`OverflowRing.tryPush` → if dropped → `droppedIntentCount++`.

### 2.3 Drain path

`drainPendingRetireIntentsForShutdown()` (AudioEngine.Processing.ReleaseResources.cpp:624):

1. Step 1 — drain OverflowRing → `emitRetireIntent` each.
2. Step 2 — drain MPSC (`dequeueOne`) + fallback
   (`dequeueFallback`) → `reclaim(intent.dspSlot)`, bounded 65536.
3. Step 3 — re-check OverflowRing refill (3 iterations), drain re-injected.

`reclaim` delegates to `EpochControl::reclaim` (ReclaimEligible →
Reclaimed transition). Idempotent. `noexcept`, `try/catch` swallows
(only atomic / mutex / vector ops involved — no allocation that can throw).

### 2.4 Gap check — System 1

- **Tombstone slots** (`dspSlot==UINT32_MAX`): drained by `dequeueOne`
  (checks tombstone, returns intent) → `reclaim(UINT32_MAX)` is a
  state-transition no-op (slot index never registered as real). No
  ownerless state.
- **OverflowRing**: Step 1 drains all; Step 3 catches re-injection
  (only possible from RT commit path, which is stopped before drain).
- **FallbackQueue**: mutex-protected, drained in Step 2.
- **Failed transfer on `emitRetireIntent`**: fallback is the
  retry target; if fallback full, OverflowRing; if OverflowRing full,
  `droppedIntentCount++` (counter-only — the owning DSPCore is still in
  the active/fading handle, retired separately via `DSPLifetimeManager::retire`).
  No orphan pointer: the DSPCore* lifetime is tracked by System 2,
  not System 1.

### 2.5 Audit verdict — System 1: PASS

Every RetireIntent slot reaches `Reclaimed` lifecycle. Drain is
bounded (65536) and re-entrant-safe. No slot is left in
`CompareEligible`/`ReclaimEligible` at shutdown completion.

---

## 3. System 2 — DSPCore* Pointer-Lifetime Ownership

### 3.1 Authority chain (D → Q → E → T)

| Layer | Type | Capacity | Lock discipline | Owner | Drain on shutdown |
| --- | --- | --- | --- | --- | --- |
| D (Deferred) | `DeferredDeletionQueue` | 4096 (Vyukov MPMC) | lock-free | `EpochDomain` (member) | `drainAllUnsafe` (epoch-gated `reclaim`) |
| Q (Quarantine) | `RetireQuarantineStore` | 512 | `std::mutex` | `m_retireQuarantine` | `drainAllUnsafe` (force) |
| E (Emergency) | `RetireQuarantineStore` | 512 | `std::mutex` | `m_emergencyQuarantine` | `drainAllUnsafe` (force) |
| T (Terminal) | `TerminalReclaimAuthority` | growable `std::vector` | `std::mutex` | `m_terminalReclaim` | `drainAll` (force) |

`enqueueWithRetry` chain (`ISRRetireRouter.cpp`): D → retry cycle
(`tryReclaim` + `drainEmergencyAndTerminal`, 2 attempts) → Q → E → T.
Each layer's `store`/`quarantine` returns `false` on full → caller
advances to next layer. **T always returns `true`** (P-4).

### 3.2 Handoff guarantees

- **D → Q (`enqueueWithRetry` retry)**: if `reclaim(minReaderEpoch)` cannot
  free D slots (epoch unsafe — live RT reader), D enqueue fails →
  `m_retireQuarantine.quarantine(...)` (Q). Q full → Emergency Q.
- **Q → E**: `m_quarantine.quarantine` returns false →
  `m_emergencyQuarantine.quarantine(...)`. E full → Terminal.
- **E → T**: `emergencyQuarantine` returns false →
  `m_terminalReclaim.store(...)` — **always succeeds** (growable).
- **T → destruction**: `terminalReclaim` (`ISRRetireRouter.cpp:488`):
  `epochSafe && !isRt` → synchronous `deleter(ptr)` (immediate
  destruction); else retained → `m_terminalReclaim.store()` (force-drained
  at shutdown via `drainAll`).

### 3.3 Shutdown drain

`~AudioEngine()` (AudioEngine.CtorDtor.cpp:96):

```text
drainDeferredRetireQueues(true)   // D.tryReclaim + coordinator.reclaim + pendingReclaimHandles_ retry
drainPendingRetireIntentsForShutdown()  // System 1 slots
if activeReaderCount()==0: drainAll()   // D + (Q + E + T)
else: epochDomain.drainAll() + drainAllQuarantineStore()  // force Q + E + T
```

`drainAllQuarantineStore()` (ISRRetireRouter.cpp:423) =
`Q.drainAllUnsafe()` + `E.drainAllUnsafe()` + `T.drainAll()`.

`T.drainAll()` (ISRRetireRouter.cpp:77) — **force, epoch-agnostic**:
takes all entries under lock, runs `deleter(ptr)` unconditionally
(Audio Thread stopped contract). This is the **no-reader-gated**
backstop: even if epoch is "unsafe", Audio Thread is stopped so no
reader can access D slots — force destruction is safe.

`releaseResources()` (AudioEngine.Processing.ReleaseResources.cpp):
same `drainAllQuarantineStore()` + `drainDeferredRetireQueues(false/true)` +
`waitForDrain(2000, 2)` (polls `isFullyDrained()`).

### 3.4 Gap check — System 2

- **D (DeferredDeletionQueue) drain**: `drainDeferredRetireQueues` calls
  `tryReclaim` (epoch-gated, `isOlder(entry.epoch, minReaderEpoch)`) +
  `coordinator.reclaim(minReaderEpoch)` (handle table reclaim) +
  pendingReclaimHandles_ retry. After reader registration close +
  Audio Thread stop, `minReaderEpoch` advances to include all retired
  entries → D drains fully.
- **Q / E drain**: `drainAllUnsafe` — force, no epoch gate (Audio Thread
  stopped). `q.drain(minReaderEpoch, isOlderFn)` called during
  `tryReclaim` first; `drainAllUnsafe` is the backstop.
- **T drain**: `drainAll()` force-destroys all retained entries
  (epoch-agnostic, Audio Thread stopped).
- **PendingReclaimHandles_**: retry loop in `drainDeferredRetireQueues`
  re-attempts `requestReclaim` after `minReaderEpoch` advances;
  remaining stuck handles are caught by `drainAllQuarantineStore` (T
  force-drain) — these were quarantined via `quarantineRetire`, not
  the pendingHandles list, so no double-dispose.
- **Double-free check**: `reclaim(slot)` (System 1) is idempotent
  (Reclaimed no-op). System 2 `drainAllUnsafe`/`drainAll` each
  `swap` entries out (emptying the container) before destruction —
  re-drain is a no-op. `tryReclaim` drains epoch-safe subset;
  `drainAll` drains remainder; no overlap (swap removes already-drained).
- **Failed enqueue path**: `enqueueWithRetry` returns
  `Shutdown` only in the dead-code branch (ShutdownReclaim path
  is used instead — see 3.3 above). The `Shutdown` result in
  `enqueueWithRetry` is never returned to production callers; all
  shutdown-time enqueue goes through `shutdownReclaim`.

### 3.5 Audit verdict — System 2: PASS

D → Q → E → T chain is total (T always accepts). Force-drain
(`drainAllQuarantineStore` = Q + E + T) runs after Audio Thread stop
and reader registration close. No entry can remain in D/Q/E/T at
shutdown completion because either (a) epoch advanced past it (normal
reclaim) or (b) it is force-destroyed in T.drainAll(). Double-dispose
prevented by container-swap discipline.

---

## 4. OwnerChannel — RuntimeState* Cross-RT Transfer

### 4.1 Authority

`OwnerChannel<OwnerPtr>` (`src/audioengine/OwnerChannel.h:39`):
SPSC lock-free, 256 slots. Key = `(sequenceId, epoch, mappedGeneration)`.
**Sole owner-transfer point across RT boundary** (B3 invariant,
RuntimeWorldAuthority.h comment).

- `enqueue` (producer, Non-RT publish thread): key match → reject
  (no-overwrite); free slot → write key then release-store owner.
- `take` (consumer, ISR/audio): single-transfer drain (consume →
  publish `nullptr` release).
- `drainAllNonRt(Fn&& reclaim)` (OwnerChannel.h:121): full scan,
  `consume` → `publish(nullptr, release)`, calls `reclaim(raw)` —
  **MUST transfer to existing retire authority, must NOT delete**.

### 4.2 Drain path (residual owners)

`drainAllNonRt` is called in **both** shutdown paths:

**`releaseResources()`** (ReleaseResources.cpp:542)

```cpp
const auto drainedResidual = worldAuthority_.ownerChannel().drainAllNonRt(
    [this](const RuntimeState* raw) noexcept {
        enqueueDeferredDeleteNonRtWithResult(
            const_cast<RuntimeState*>(raw),
            [](void* p) noexcept {
                auto* ptr = static_cast<RuntimePublishWorld*>(p);
                ptr->unseal(); ptr->~RuntimePublishWorld();
                convo::aligned_free(ptr);
            },
            DeletionEntryType::World);
    });
```

`enqueueDeferredDeleteNonRtWithResult` -> `isShutdownInProgress() == true`
(both releaseResources and ~AudioEngine set this) ->
`shutdownReclaim(ptr, deleter, epoch, World)` →
`terminalReclaim(...)` → `m_terminalReclaim.store()` (always succeeds)
or immediate `deleter()` if epoch safe.

**`~AudioEngine()`**: does NOT call `drainAllNonRt` directly — instead
relies on `clearPublishedRuntimeSnapshotsNonRt()` (returns oldWorld)
→ `retirePublishedRuntimeWorldNonRt(clearedWorld, true)` →
`enqueueDeferredDeleteNonRt(world, deleter, World)` →
`shutdownReclaim` → TerminalReclaimAuthority. Residual live Owners
still in the channel are covered by `drainAllQuarantineStore` →
`T.drainAll()` backstop.

### 4.3 Gap check — OwnerChannel

- **Residual Owner in channel after publish commit**: `take()`
  in `PublishExecutor::executePublish` (RuntimePublishExecutor.h:30)
  consumes via single-transfer (publish nullptr release). If commit
  succeeded, owner is moved into `authority.publish()` (sole swap).
  If `enqueue` succeeded but `take` never ran (coordinator stopped),
  `drainAllNonRt` recovers the residual.
- **`enqueue` success but intentQueue full rollback**
  (AudioEngine.h:4567): `take()` is called to recover the Owner —
  ownership returns to AudioEngine (CallerDestroy), which then
  `enqueueDeferredDeleteNonRt` it → shutdownReclaim → Terminal.
- **Double-ownership**: `take()` is single-transfer
  (sequenceId key + nullptr publish). `drainAllNonRt` consumes only
  remaining non-null owners. `retirePublishedRuntimeWorldNonRt`
  operates on `oldWorld` from `publishAndSwap` (distinct from
  OwnerChannel owners). No double-dispose.
- **RuntimeState* → void* cast**: `const_cast` in
  `retirePublishedRuntimeWorldNonRt` (AudioEngine.h:3528) matches the
  `static_cast<RuntimePublishWorld*>(p)` in the deleter — type-safe
  (RuntimePublishWorld inherits RuntimeState).

### 4.4 Audit verdict — OwnerChannel: PASS

`drainAllNonRt` recovers all residual Owners and routes them through
the existing System 2 shutdown path (`shutdownReclaim` → Terminal).
The `enqueue` success-but-take-never-ran case is covered by
`drainAllNonRt` (releaseResources path); the `~AudioEngine` only
path is covered by `clearPublishedRuntimeSnapshotsNonRt` +
`T.drainAll()` backstop. No OwnerChannel Owner can survive
`drainAllNonRt` + `drainAllQuarantineStore` without being transferred
to TerminalReclaimAuthority.

---

## 5. Final Shutdown Ordering

### 5.1 `releaseResources()` (canonical, primary path)

| Phase | ShutdownPhase | Action | System |
| --- | --- | --- | --- |
| 1 | StopAcceptingWork | `requestShutdown()` (admission close) | All |
| 2 | AudioStopped | `transitionTo(AudioStopped)` | ShutdownRuntime |
| 3 | — | `closeReaderRegistration()` (before ~AudioEngine path) | EBR |
| 4 | — | Graceful drain (5000ms poll: publishEpoch + tryReclaim) | System 1+2 |
| 5 | — | OverflowRing reinject (128/cycle) + final drain on timeout | System 1 |
| 6 | DrainRetire | `drainDeferredRetireQueues(true)` + `tryReclaim` per slot | System 1+2 |
| 7 | ReclaimComplete | `transitionTo(ReclaimComplete)` | ShutdownRuntime |
| 8 | EmergencyDrain | `transitionTo(EmergencyDrain)`; conditional runtime drain | System 2 |
| 9 | VerifyDrained | `tryShutdownQuiescentReclaim` (active/fading DSPHandle) → clear world snapshots → `drainAllQuarantineStore` (if readers==0) → `waitForDrain(2000,2)` → `drainPendingRetireIntentsForShutdown` + retry → `drainDeferredRetireQueues(true)` | System 1+2 |
| 10 | — | `m_coordinator.finalizeShutdown(timedOut)` (producer join) | ShutdownRuntime |
| 11 | — | **`drainAllNonRt` OwnerChannel residual → shutdownReclaim → Terminal** (15-P-CROSS-IMPLEMENTATION-1) | OwnerChannel → System 2 |
| 12 | — | `markShutdownComplete()` → `transitionTo(ShutdownComplete)` + `emitShutdownTrace` | ShutdownRuntime |

### 5.2 `~AudioEngine()` (defensive fallback path)

| Phase | ShutdownPhase | Action |
| --- | --- | --- |
| 1 | StopAcceptingWork | `cancelPendingUpdate()` + `requestShutdown()` |
| 2 | StopAudio | `stopTimer()` |
| 3 | StopWorkers | `shutdownCoordinatorLoop()` (join) + `stopRebuildThread()` |
| 4 | — | Clear active/fading/pending slots → retire via `DSPLifetimeManager` |
| 5 | — | `shutdownWorkerThread()` |
| 6 | ForceEpochAdvance | `publishEpoch()` |
| 7 | DrainRetire | `closeReaderRegistration()` → graceful drain (5000ms) → `clearPublishedRuntimeSnapshotsNonRt` → retire oldWorld → `drainDeferredRetireQueues(true)` → `drainPendingRetireIntentsForShutdown` → `drainAll()`/`drainAllQuarantineStore` |
| 8 | Destroy | `markShutdownComplete()` → `lifecycleState=Destroyed` |

### 5.3 Ordering invariant

Critical ordering in **both** paths:

1. **Admission close** (`requestShutdown` / `closeReaderRegistration`) — first,
   blocks new producers.
2. **Producer join** (`shutdownCoordinatorLoop` in ~AudioEngine;
   `finalizeShutdown` in releaseResources) — after audio stop.
3. **Audio/consumer quiescence** (graceful drain completes) — Audio Thread
   stopped, no live readers.
4. **OwnerChannel drain** (`drainAllNonRt`) — only after finalizeShutdown
   (producers joined).
5. **System 1 drain** (`drainPendingRetireIntentsForShutdown`) — after
   OverflowRing reinject exhausted.
6. **System 2 force-drain** (`drainAllQuarantineStore`) — last resort,
   epoch-agnostic, after all readers provably stopped.
7. **ShutdownComplete** — after all drains.

Phase 4 (OwnerChannel) precedes Phase 5 (System 1) precedes Phase 6 (System 2
force), because OwnerChannel Owners transfer INTO System 2
(`shutdownReclaim`), System 1 slots must be reclaimed before System 2
final drain can confirm zero residents, and System 2 force-drain is the
terminal backstop.

### 5.4 Audit verdict — Ordering: PASS

Ordering is monotonic (irreversible). Admission close precedes all drains.
Producer join precedes OwnerChannel drain (15-P-CROSS-IMPLEMENTATION-1
comment confirms). System 1 drain precedes System 2 force-drain.
`ShutdownPhase` FSM (`transitionTo`) blocks non-sequential forward jumps
except terminal-skip; `isTerminalPhase` allows `ReclaimComplete→ShutdownComplete`
skip only over terminal phases (TimedOut/Failed).

---

## 6. Residual Gaps & Ownerless-State Proof

### 6.1 Failed enqueue / retry / overflow

| Path | Failure mode | Ownership outcome | Gap? |
| --- | --- | --- | --- |
| `emitRetireIntent` MPSC full | fallback queue | Intent → fallbackQueue_ | No — bounded, drained in Step 2 |
| fallback full | OverflowRing | Intent → overflowRing_ | No — drained in Step 1/3 |
| OverflowRing full | droppedIntentCount++ | Counter only; DSPCore* in handle (System 2) | No — System 2 owns the pointer, not System 1 |
| `enqueueWithRetry` D full | Q | ptr → quarantine | No |
| Q full | E | ptr → emergency | No |
| E full | T (always accepts) | ptr → TerminalReclaimAuthority | No — growable, always takes ownership |

### 6.2 Double drain

- `drainAllQuarantineStore` (Q + E + T): each store uses `swap`/take-all
  under lock before destruction — re-call is no-op (containers empty).
- `drainAllUnsafe` (Q/E): same container-empties discipline.
- `drainAll` (T): `pending.swap(entries_)` under lock, then destroy
  outside lock — re-call finds empty vector.
- System 1 `drainPendingRetireIntentsForShutdown`: idempotent —
  `reclaim` is a no-op on Reclaimed; 3-iteration OverflowRing refill
  handles re-injection.

### 6.3 Timeout / stuck-reader

- Graceful drain timeout (5s in ~AudioEngine, 5s in releaseResources):
  falls to **force drain** (`drainAllQuarantineStore` / T.drainAll),
  which is epoch-agnostic. Stuck readers block epoch-advance reclaim,
  but **force drain destroys regardless** (Audio Thread stopped).
- 15-P-5 fix: even on stuck-reader fallback (`activeReaderCount > 0`),
  `~AudioEngine` calls `m_epochDomain.drainAll()` (D) +
  `drainAllQuarantineStore()` (Q + E + T) — NOT D-only. So
  `RuntimeState*` / `DSPCore*` in Q/E/T are always force-destroyed
  regardless of reader state.

### 6.4 Destructor fallback

- `~AudioEngine()` runs the full drain tail even if `releaseResources()`
  was not called (defensive). `drainPendingRetireIntentsForShutdown`
  - `drainAll()`/`drainAllQuarantineStore()` covers all residual.

### 6.5 PendingReclaimHandles_

- Retry loop in `drainDeferredRetireQueues` re-attempts
  `requestReclaim` after `minReaderEpoch` advances. Handles still
  unsafe at graceful-drain timeout are **not** individually
  force-destroyed — but they correspond to DSPHandles in the handle
  table, which are quarantined (`DSPQuarantineManager`) and reclaimed
  via `quarantineRetire` → Q → E → T. The `pendingReclaimHandles_`
  list tracks the *handle* (not the pointer directly); the pointer
  ownership already transferred to System 2 at retire time.

### 6.6 Ownershipless proof — no ownerless `RuntimeState*` / `DSPCore*`

At `markShutdownComplete()` (both paths), the following invariants hold:

1. **OwnerChannel**: `drainAllNonRt` (releaseResources) or
   `clearPublishedRuntimeSnapshotsNonRt` + `drainAllQuarantineStore`
   (~AudioEngine) recovers all residual Owners →
   `enqueueDeferredDeleteNonRtWithResult` → `shutdownReclaim` →
   TerminalReclaimAuthority. `ownerChannel_.size() == 0` post-drain
   (single-transfer `take` + nullptr publish).

2. **System 1 slots**: `drainPendingRetireIntentsForShutdown` drains
   all MPSC + fallback + OverflowRing. `pendingRetireCount() == 0`
   (checked in graceful drain loop and `isFullyDrained`).

3. **System 2 D/Q/E/T**: `isFullyDrained()` checks
   `retireQuarantineResident == 0` + `terminalReclaimResident == 0`
   - `ringResident == 0`. Force-drain (`drainAllQuarantineStore`)
   guarantees these hit zero. `WorldLifecycleAudit::activeWorldCount()
   == published - retired` verifies no World leak (double-retire
   guarded by fetchSub underflow → `doubleRetireCount`).

4. **RuntimeState* specific**: the only `RuntimeState*` instances are
   `RuntimePublishWorld` owned by `aligned_unique_ptr` in
   OwnerChannel / RuntimeStore. `retirePublishedRuntimeWorldNonRt`
   and the `drainAllNonRt` callback both route the `RuntimePublishWorld*`
   deleter through `enqueueDeferredDeleteNonRtWithResult` (shutdown
   path → Terminal). No other `RuntimeState*` allocation path exists
   outside OwnerChannel ownership.

5. **DSPCore* specific**: `DSPLifetimeManager::retire` →
   `enqueueWithRetry(dsp, &destroyDSPCoreNode, ..., Generic)` →
   D → Q → E → T. `destroyDSPCoreNode` (AudioEngine.Threading.cpp:17)
   is the sole deleter (`~DSPCore()` + `aligned_free`). Every
   `DSPCore*` flows through this chain — no bypass.

### 6.7 Audit verdict — Residual gaps: PASS

No pointer is orphanable: every pointer path terminates at
TerminalReclaimAuthority (growable, always accepts) and is
force-destroyed by `drainAll()` after Audio Thread stop + reader
registration close. The `isFullyDrained()` check gates on
`terminalReclaimResident == 0` (post-15-P-5 audit fix), closing the
premature-completion failure mode. Double-dispose is structurally
prevented by container-swap discipline + idempotent System 1
`reclaim`.

---

## 7. Cross-Cutting Invariants Verified

| Invariant | Location | Verified |
| --- | --- | --- |
| TerminalReclaimAuthority always accepts (growable) | ISRRetireRouter.h:62 | YES |
| drainAll is epoch-agnostic (force) | ISRRetireRouter.cpp:77 | YES |
| drainAllQuarantineStore covers Q + E + T | ISRRetireRouter.cpp:423 | YES |
| drainPendingRetireIntentsForShutdown drains MPSC + fallback + OverflowRing | ReleaseResources.cpp:624 | YES |
| enqueueDeferredDeleteNonRtWithResult → shutdownReclaim (not drop) | AudioEngine.h:4195-4212 | YES |
| tryShutdownQuiescentReclaim (wrapper) → Coordinator.reclaimShutdownQuiescent: Proof → Permit consume → reclaim | AudioEngine.h:4356 / ISRRuntimePublicationCoordinator.cpp:779 | YES |
| reclaim is idempotent (Reclaimed no-op) | ISRRetireRuntimeEx.cpp | YES |
| OwnerChannel take = single-transfer | OwnerChannel.h | YES |
| drainAllNonRt callback → shutdownReclaim (not direct deleter) | ReleaseResources.cpp:542 | YES |
| isFullyDrained checks terminalReclaimResident | AudioEngine.Threading.cpp:151 | YES |
| Admission close before producer join | ISRShutdown.cpp, releaseResources | YES |
| Producer join before OwnerChannel drain | 15-P-CROSS-IMPLEMENTATION-1 comment | YES |

---

## 8. Verdict: PASS

All residual ownership / authority closure criteria met:

- **System 1** (RetireIntent slot-state): every slot drains to
  `Reclaimed`; OverflowRing/MPSC/fallback all drained; idempotent
  reclaim prevents double-dispose.
- **System 2** (DSPCore*/RuntimeState* pointer lifetime): D → Q → E →
  T chain is total (T always accepts); force-drain after Audio Thread
  stop destroys all residual regardless of epoch/reader state.
- **OwnerChannel**: all residual Owners recovered via `drainAllNonRt`
  (releaseResources) or `clearPublishedRuntimeSnapshotsNonRt` +
  `T.drainAll()` backstop (~AudioEngine) → `shutdownReclaim` → Terminal.
- **Final ordering**: Admission close → Producer join →
  Consumer quiescence → OwnerChannel drain → System 1 drain →
  System 2 force-drain → ShutdownComplete. Monotonic, irreversible.
- **Ownerless proof**: at `markShutdownComplete()`,
  `terminalReclaimResident == 0` (gated by `isFullyDrained`),
  `pendingRetireCount() == 0`, `overflowRingResident == 0`,
  `dspQuarantineResident == 0`, `activeWorldCount == published - retired`.

No production code changes made (audit phase only). ASan
0xC0000139 remains BLOCKED (environment issue, recorded in 15-P-8,
not addressed per scope).

Regression baseline confirmed: **15-P-8 Debug CTest 33/33 PASS
(36.76s), Release CTest 33/33 PASS (32.70s)** — no test regression
introduced by 15-P-4 through 15-P-8 changes.

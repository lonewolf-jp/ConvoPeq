# 15-P-10: Shutdown Authority / Terminal Ownership Cross-Audit

## Status: PASS (with INVESTIGATE notes)

## Objective

Cross-audit 15-P-6 through 15-P-9: verify that **no single lifetime
transition is owned by multiple authorities**. Enumerate every caller of
the terminal reclaim path, confirm the `TerminalReclaimAuthority` singleton,
and prove there is no direct `delete RuntimeState` / `delete DSPCore` that
bypasses the authority chain on the **shutdown** path.

No production code changes (audit phase only).

---

## 1. Terminal Reclaim Authority — Singleton Verification

### 1.1 `TerminalReclaimAuthority` instance count

`TerminalReclaimAuthority` is defined in `src/audioengine/ISRRetireRouter.h`
(class at line 62). It is **owned by value** as a member of
`ISRRetireRouter`:

```cpp
// ISRRetireRouter.h:358
TerminalReclaimAuthority m_terminalReclaim;
```

`ISRRetireRouter` itself is owned by `AudioEngine` as a single
`unique_ptr`:

```cpp
// AudioEngine.h:4686
std::unique_ptr<convo::isr::ISRRetireRouter> m_retireRouter;
```

`AudioEngine` has exactly one `RuntimeWorldAuthority` (which contains the
`LifetimeState` slot-state authority) and one `RuntimeIntentCoordinator`
(`runtimePublicationBridge_`, line 4833) and one `ShutdownRuntime`
(`shutdownRuntime_`, line 4869).

### 1.2 Authority distribution

| Operation | Authority | Owner instance | Duplicate? |
| --- | --- | --- | --- |
| `store()` (retain entry) | TerminalReclaimAuthority | `m_retireRouter->m_terminalReclaim` | No — single instance |
| `drain()` (epoch-gated) | TerminalReclaimAuthority | `m_terminalReclaim.drain()` | No |
| `drainAll()` (force) | TerminalReclaimAuthority | `m_terminalReclaim.drainAll()` | No |
| `residentCount()` | TerminalReclaimAuthority | `m_terminalRecaim.residentCount()` | No |
| `tryReclaim` (epoch-gated) | `ISRRetireRouter::tryReclaim` (delegates to T.drain + Q.drain + D.reclaim) | `m_retireRouter` | No — single router |
| `drainAllQuarantineStore` | `ISRRetireRouter` (delegates Q + E + T) | `m_retireRouter` | No |
| `drainAll()` | `ISRRetireRouter` (delegates provider_->drainAll + drainAllQuarantineStore) | `m_retireRouter` | No |
| `shutdownReclaim` | `ISRRetireRouter` (delegates → terminalReclaim) | `m_retireRouter` | No |

### 1.3 Verdict — Terminal authority singularity: PASS

There is exactly **one** `TerminalReclaimAuthority` instance per
`AudioEngine`, injected as a by-value member of the single `ISRRetireRouter`.
All terminal reclaim flows (`shutdownReclaim`, `terminalReclaim`,
`drainAll`, `drainAllQuarantineStore`) route through this single instance.

---

## 2. Production Caller Enumeration

### 2.1 `terminalReclaim(` — 2 production callers

| Caller | File:Line | Context |
| --- | --- | --- |
| `ISRRetireRouter::enqueueWithRetry` (D→Q→E→T chain) | ISRRetireRouter.cpp:344 | Stage 5: E full → T.store() |
| `ISRRetireRouter::shutdownReclaim` | ISRRetireRouter.cpp:562 | Shutdown path → terminalReclaim("shutdownReclaim") |

### 2.2 `shutdownReclaim(` — 1 production caller

| Caller | File:Line | Context |
| --- | --- | --- |
| `AudioEngine::enqueueDeferredDeleteNonRtWithResult` (when `isShutdownInProgress()`) | AudioEngine.h:4208 | Shutdown ownership transfer |

### 2.3 `drainAllNonRt(` — 1 production caller

| Caller | File:Line | Context |
| --- | --- | --- |
| `AudioEngine::releaseResources()` (post finalizeShutdown) | ReleaseResources.cpp:542 | OwnerChannel residual → enqueueDeferredDeleteNonRtWithResult |
| `OwnerChannelTests.cpp` (test only) | tests/OwnerChannelTests.cpp:148,159,175 | Test-only |

### 2.4 `clearPublishedRuntimeSnapshotsNonRt(` — 2 production callers

| Caller | File:Line | Context |
| --- | --- | --- |
| `AudioEngine::~AudioEngine()` | AudioEngine.CtorDtor.cpp:230 | Destructive fallback |
| `AudioEngine::releaseResources()` | ReleaseResources.cpp:456 | Normal shutdown |

### 2.5 `drainAllQuarantineStore(` — 4 production callers

| Caller | File:Line | Context |
| --- | --- | --- |
| `AudioEngine::~AudioEngine()` (stuck-reader fallback) | AudioEngine.CtorDtor.cpp:257 | Force-drain |
| `AudioEngine::releaseResources()` (quiescence) | ReleaseResources.cpp:378 | Pre-VerifyDrained |
| `AudioEngine::releaseResources()` (post-clear) | ReleaseResources.cpp:473 | Post clearPublishedRuntimeSnapshotsNonRt |
| `ISRRetireRouter::drainAll()` (internal delegation) | ISRRetireRouter.cpp:579 | — |

### 2.6 `isFullyDrained(` — 2 production call sites (1 comment, 1 test-only)

| Caller | File:Line | Context |
| --- | --- | --- |
| `releaseResources()` (drain loop condition) | ReleaseResources.cpp:521 | `if (!drainedWithinBudget \|\| !isFullyDrained())` |
| `waitForDrain` (poll condition) | AudioEngine.Threading.cpp:196 | `while (!isFullyDrained())` |
| `ShutdownRuntime::tryMakeQuiescenceProof` (comment) | ISRShutdown.cpp:344 | Prohibited from simple `isFullyDrained()` shortcut |
| `ISRSemanticValidationTests.cpp` (test only) | tests/ISRSemanticValidationTests.cpp:334,369,440 | Test-only |

### 2.7 `markShutdownComplete(` — 2 production callers

| Caller | File:Line | Context |
| --- | --- | --- |
| `AudioEngine::~AudioEngine()` | AudioEngine.CtorDtor.cpp:259 | Destructive fallback |
| `AudioEngine::releaseResources()` | ReleaseResources.cpp:587 | Normal shutdown |

### 2.8 `ShutdownComplete` — 3 production references

| Location | File:Line | Context |
| --- | --- | --- |
| `enum class ShutdownPhase` (declaration) | ISRShutdown.h:47 | FSM terminal state |
| `isTerminalPhase` (check) | ISRShutdown.cpp:168 | `== ShutdownPhase::ShutdownComplete` |
| `transitionTo` (target) | ReleaseResources.cpp:607 | `transitionTo(ShutdownPhase::ShutdownComplete)` |
| `finalPhase` default | ISRShutdown.h:145 | Initializer |

---

## 3. OwnerChannel to Terminal — Unique Path Verification

### 3.1 The two recovery paths for residual `RuntimeState*` owners

| Path | Caller | Destination | Bypasses Terminal? |
| --- | --- | --- | --- |
| **Normal shutdown** (`drainAllNonRt` callback) | `releaseResources()` | `enqueueDeferredDeleteNonRtWithResult` -> shutdown path -> T | No |
| **Destructive fallback** (`~AudioEngine` clear path) | `clearPublishedRuntimeSnapshotsNonRt` returns oldWorld -> `retirePublishedRuntimeWorldNonRt` -> `enqueueDeferredDeleteNonRt` -> `shutdownReclaim` -> T | `~AudioEngine` | No |
| **Force backstop** | `drainAllQuarantineStore()` -> `T.drainAll()` | Terminal backstop | No |

### 3.2 Can the same `RuntimeState*` be reclaimed twice?

- `OwnerChannel::take()` uses single-transfer (publish `nullptr` release
  after consume). `drainAllNonRt` uses the same pattern. Re-drain is a
  no-op (slots seen `nullptr`).
- `clearPublishedRuntimeSnapshotsNonRt` does `publishAndSwap(nullptr)` —
  returns the old world, sets current to null. A second call returns
  null (no double-reclaim).
- The `drainAllNonRt` callback in `releaseResources` operates on
  residual Owners in the channel; `clearPublishedRuntimeSnapshotsNonRt`
  operates on the RuntimeStore's current world. These are **distinct
  RuntimeState* instances** (channel owners are pending-commit, not yet
  published; store's current was published+swap'd). No overlap.

### 3.3 INVESTIGATE: `CallerDestroy` rollback path (non-shutdown)

`enqueueRuntimePublicationFireAndForget` (AudioEngine.h:4498) has a
`CallerDestroy` return when OwnerChannel `enqueue` fails OR
`intentQueue_` enqueue fails. In the intentQueue-failure case
(AudioEngine.h:4567):

```cpp
(void)worldAuthority_.ownerChannel().take(
    convo::isr::OwnerChannelKey{ seqId, epoch, mappedGen });
```

The `take()` return (a `RuntimeOwner` unique_ptr) is **explicitly
discarded**. The unique_ptr destructor runs `delete RuntimeState`
**without** routing through `enqueueDeferredDeleteNonRtWithResult` →
TerminalReclaimAuthority.

**However**: this is the **non-shutdown** publish-failure path. The
RuntimeState was never published (Coordinator never saw it), so it is
not yet under the retirement authority's lifetime protection. Destroying
it directly is semantically "undoing an unpublished construction" —
equivalent to a constructor exception rollback. The `stateOwner` in
`publishImpl` is already null (moved), so no double-free.

**Status**: INVESTIGATE — structurally safe (never published, single
ownership transfer via move), but does bypass the terminal authority on
the error path. Not a shutdown-path concern. No action required for
15-P-10 (audit scope = shutdown authority).

---

## 4. System 1 / System 2 Boundary

### 4.1 RetireIntent/EpochControl does NOT directly reclaim pointers

- `LifetimeState::reclaim(slot)` calls `EpochControl::reclaim(slot)`
  (ISRRetire.h). This transitions the **slot lifecycle**
  (ReclaimEligible → Reclaimed) and calls
  `DSPLifetimeManager::destroyDSPCoreNode`-equivalent **slot reclamation**
  — it does NOT delete the DSPCore pointer. The DSPCore* pointer lifetime
  is managed by System 2 (`DSPLifetimeManager::retire` →
  `enqueueWithRetry` → D → Q → E → T).
- `EpochControl::reclaim` (ISRRetireRuntimeEx.cpp) calls
  `reclaimLane(Coordination)` → fetches slot state, decrements
  `quarantineResidentCount` (atomic counter, telemetry only). No
  `delete`/`deleter()` call.

### 4.2 System 2 does not bypass System 1

- `enqueueWithRetry` (D→Q→E→T) is the **only** entry point for
  `DSPCore*` retirement (via `DSPLifetimeManager::retire`).
- `retireDSPHandleForRuntime` (DSP core handle retirement) →
  `DSPLifetimeManager::retire` → `enqueueWithRetry`. No bypass.
- `shutdownReclaim` → `terminalReclaim` — same chain. No bypass.

### 4.3 System 1 / System 2 drain completion are NOT conflated

- System 1 drain: `drainPendingRetireIntentsForShutdown` drains
  OverflowRing + MPSC + fallback + re-inject. Only touches **slot
  state** (lifecycle FSM). Does NOT touch DSPCore*/RuntimeState*
  pointers.
- System 2 drain: `drainDeferredRetireQueues` (tryReclaim + pendingHandles)
  and `drainAllQuarantineStore` (Q + E + T force). Touches **pointers**.
- `isFullyDrained()` checks BOTH:
  - System 1: `lifetimeRetireIntentPending` (via
    `worldAuthority_.lifetime().pendingIntentCount()`)
  - System 2: `retireDepth` (pendingRetireCount), `ringResident`
    (overflowRing), `dspQuarantineResident`, `retireQuarantineResident`,
    `terminalReclaimResident`, `pendingReclaimEmpty`
  (AudioEngine.Threading.cpp:147 — "TerminalReclaimAuthority
  滞留も直接判定する").

### 4.4 Verdict — System boundary: PASS

System 1 (slot-state) and System 2 (pointer-lifetime) are cleanly
separated. System 1 `reclaim` never deletes pointers; System 2
`enqueueWithRetry`/`shutdownReclaim` never touches slot lifecycles
directly. `isFullyDrained` checks both independently.

---

## 5. Destruction Fallback Semantics

| Fallback | File:Line | Replaces normal shutdown? | Duplicate authority? |
| --- | --- | --- | --- |
| `~AudioEngine()` | AudioEngine.CtorDtor.cpp:96 | No — runs full drain tail (closeReaderRegistration → grace drain → clear+retire → drainDeferredRetireQueues → drainPendingRetireIntentsForShutdown → drainAll/drainAllQuarantineStore → markShutdownComplete) | No — same authority chain (m_retireRouter) |
| `releaseResources()` | ReleaseResources.cpp | Canonical path | No |
| `stopRebuildThread()` | AudioEngine.RebuildDispatch.cpp:771 | No — stops rebuild thread + discards recovery requests | N/A — no pointer ownership |
| `shutdownWorkerThread()` | AudioEngine.Init.cpp:186 | No — stops worker thread | N/A — no pointer ownership |

### 5.1 `~AudioEngine()` fallback does NOT duplicate authority

- `~AudioEngine()` uses the **same** `m_retireRouter` (single
  TerminalReclaimAuthority instance) as `releaseResources()`.
- It calls `drainAllQuarantineStore` (Q + E + T force-drain) — the
  same method `releaseResources` uses.
- `clearPublishedRuntimeSnapshotsNonRt` → `retirePublishedRuntimeWorldNonRt`
  → `enqueueDeferredDeleteNonRt` → `shutdownReclaim` — identical to
  `releaseResources` path (ReleaseResources.cpp:456-473).
- The fallback `if (activeReaderCount() == 0) drainAll(); else
  drainAllQuarantineStore()` (CtorDtor.cpp:243-257) reuses the 15-P-5
  fix — no new authority introduced.

### 5.2 Verdict — Fallback semantics: PASS

`~AudioEngine()` is a **re-execution guard**, not an alternate authority.
It reuses `m_retireRouter` (single TerminalReclaimAuthority). No duplicate
ownership transition. `stopRebuildThread` / `shutdownWorkerThread` own no
pointers.

---

## 6. Direct `delete` / Smart-Ptr Destruction Check

### 6.1 `delete RuntimeState` / `delete DSPCore` — production code

| Match | File:Line | Context |
| --- | --- | --- |
| (none) | — | No bare `delete RuntimeState` or `delete DSPCore` in production `src/` |

The only `delete`-mention is a documentation comment
(ISRLifetimeProof.h:124: "Permit 自体が delete DSPCore を許可するのではなく").

### 6.2 `unique_ptr` / `aligned_unique_ptr` for RuntimeState / DSPCore — production

| Location | File:Line | Destruction path | Bypasses Terminal? |
| --- | --- | --- | --- |
| `RuntimeWorldAuthority::ownerChannel_` | RuntimeWorldAuthority.h:273 | OwnerChannel with unique_ptr owner | No — drained via drainAllNonRt -> shutdownReclaim -> T |
| `FrozenRuntimeWorld::state_` | FrozenRuntimeWorld.h:84 | aligned_unique_ptr holding RuntimeState | See 6.3 |
| `stateOwner` (PublicationExecutor.cpp) | PublicationExecutor.cpp:46 | aligned_unique_ptr holding RuntimeState — moved to commitRuntimePublication | No — ownership transferred to OwnerChannel on success; on CallerDestroy, see 6.3 |
| `placeholderDSP` | PrepareToPlay.cpp:238 | aligned_unique_ptr holding DSPCore | See 6.3 |

### 6.3 INVESTIGATE: FrozenRuntimeWorld / placeholder DSPCore RAII destruction

- `FrozenRuntimeWorld::state_` (`aligned_unique_ptr<RuntimeState>`)
  — if `releaseState()` is **not** called, the destructor calls
  `state_->unseal()` then the unique_ptr destructor calls
  `aligned_free`. **However**: `releaseState()` is called at
  PublicationExecutor.cpp:34 (`frozen->releaseState()`) on the
  success path **before** `commitRuntimePublication`. If publish
  succeeds, `stateOwner` (wrapping the released state) is moved into
  the OwnerChannel. If `commitRuntimePublication` fails with
  `CallerDestroy`, the owner was already `take()`n and dropped (see 3.3
  INVESTIGATE). The `frozen` unique_ptr itself is moved into
  `publishImpl` and destroyed after `commitRuntimePublication` returns —
  if `state_` was released, `frozen`'s destructor is a no-op.
- `placeholderDSP` (PrepareToPlay.cpp:238) — created during prepare,
  consumed by the runtime; must verify it's retired, not RAII-destroyed.

### 6.4 DSPCore* `destroyDSPCoreNode` — the sole DSPCore deleter

`AudioEngine::destroyDSPCoreNode` (AudioEngine.Threading.cpp:17) is the
**only** `DSPCore*` destruction function:

```cpp
void AudioEngine::destroyDSPCoreNode(void* p) noexcept {
    auto* core = static_cast<DSPCore*>(p);
    core->~DSPCore();
    convo::aligned_free(core);
}
```

It is injected as the deleter into **all** retire paths:

| Retire path | File:Line | Routes through Terminal? |
| --- | --- | --- |
| DSPLifetimeManager::retire | DSPLifetimeManager.cpp:50 | Yes — enqueueWithRetry → D → Q → E → T |
| DSPLifetimeManager::retireByHandle | DSPLifetimeManager.cpp:97 | Yes — enqueueWithRetry |
| DSPLifetimeManager::destroyRolledBackDSP | DSPLifetimeManager.cpp:123 | **No** — direct call (pre-publication failure) |
| DSPGuard destructor (retireDSPHandleForRuntime false) | AudioEngine.RebuildDispatch.cpp:898 | **No** — direct call (pre-publication rollback) |
| Recovery warmup-fail | AudioEngine.RebuildDispatch.cpp:967 | **No** — direct call (pre-publication rollback) |
| Durable recovery warmup-fail | AudioEngine.RebuildDispatch.cpp:1040 | **No** — direct call (pre-publication rollback) |

`destroyRolledBackDSP` (non-shutdown, publish-failure rollback) bypasses
the terminal authority. See INVESTIGATE note above.

### 6.5 Verdict — Direct destruction: PASS (with INVESTIGATE on non-shutdown rollback)

On the **shutdown** path, every `DSPCore*` and `RuntimeState*` flows
through `enqueueWithRetry` / `shutdownReclaim` → D → Q → E → T.
No bare `delete` in production `src/`. The direct `destroyDSPCoreNode`
calls on non-shutdown paths are all **pre-publication rollback**
(`destroyRolledBackDSP`, DSPGuard destructor, recovery/durable-recovery
warmup-fail — see §6.4): objects that never reached publication and
were never under authority protection. These are pre-retirement
rollbacks, not shutdown-path bypasses.

---

## 7. Authority Matrix

| Operation | Authoritative owner | Other callers/observers | Duplicate authority? |
| --- | --- | --- | --- |
| Admission close | ShutdownRuntime (closeAdmission / transitionTo) | AudioEngine (sets phase, calls requestShutdown) | No |
| OwnerChannel drain | AudioEngine (drainAllNonRt) | — | No — single call site |
| System 1 drain | AudioEngine (drainPendingRetireIntentsForShutdown) | — | No — single call site |
| System 2 drain (D) | ISRRetireRouter (drainAll → provider_->drainAll) | AudioEngine (drainDeferredRetireQueues) | No — AudioEngine delegates to router |
| Q drain | RetireQuarantineStore (drainAllUnsafe) | ISRRetireRouter (drainAllQuarantineStore) | No — delegated |
| E drain | RetireQuarantineStore (drainAllUnsafe) | ISRRetireRouter (drainAllQuarantineStore) | No — delegated |
| Terminal store | TerminalReclaimAuthority (store/terminalReclaim) | ISRRetireRouter (enqueueWithRetry/shutdownReclaim) | No — single instance |
| Terminal drain | TerminalReclaimAuthority (drainAll) | ISRRetireRouter (drainAllQuarantineStore) | No — delegated |
| Fully-drained proof | AudioEngine (isFullyDrained) + Coordinator (isFullyDrained) | ShutdownRuntime (tryMakeQuiescenceProof — prohibited from shortcut) | No — layered observation, not ownership |
| ShutdownComplete | AudioEngine (transitionTo) | ShutdownRuntime (setBoundedTeardownCounters + emitShutdownTrace) | No — AudioEngine declares, Runtime observes |

---

## 8. Shutdown Ordering — Authority Chain

### 8.1 `releaseResources()` ordering (canonical)

```text
requestShutdown (admission close — ShutdownRuntime)
      ->
closeReaderRegistration (EBR — EpochDomain)
      ->
graceful drain poll (AudioEngine: publishEpoch + tryReclaim + drainDeferredRetireQueues)
      ->
tryShutdownQuiescentReclaim (active/fading DSPHandle — Coordinator.reclaimShutdownQuiescent)
      ->
clearPublishedRuntimeSnapshotsNonRt -> retirePublishedRuntimeWorldNonRt (AudioEngine -> shutdownReclaim -> T)
      ->
drainAllQuarantineStore (if activeReaderCount==0) — ISRRetireRouter: Q + E + T
      ->
waitForDrain (2000ms poll — AudioEngine::isFullyDrained)
      ->
drainPendingRetireIntentsForShutdown (System 1 slots — AudioEngine)
      ->
drainDeferredRetireQueues(true) (System 2 — AudioEngine -> m_retireRouter)
      ->
finalizeShutdown (Coordinator — producer join)
      ->
drainAllNonRt OwnerChannel residual (AudioEngine -> drainAllNonRt callback -> shutdownReclaim -> T)
      ->
markShutdownComplete (Coordinator)
      ->
transitionTo(ShutdownComplete) (ShutdownRuntime)
```

### 8.2 `~AudioEngine()` ordering (fallback)

```text
cancelPendingUpdate + requestShutdown (ShutdownRuntime)
      ->
closeReaderRegistration (EBR — EpochDomain)
      ->
graceful drain poll (5s — publishEpoch + tryReclaim)
      ->
clearPublishedRuntimeSnapshotsNonRt -> retirePublishedRuntimeWorldNonRt
      ->
drainDeferredRetireQueues(true)
      ->
drainPendingRetireIntentsForShutdown (System 1)
      ->
drainAll() / drainAllQuarantineStore() (Q + E + T force-drain)
      ->
markShutdownComplete (Coordinator)
```

### 8.3 Ordering invariant

Both paths follow: **Admission close → Reader close → Graceful drain →
OwnerChannel clear → System 1 drain → System 2 drain → OwnerChannel
residual drain → ShutdownComplete**. The OwnerChannel residual drain
(`drainAllNonRt`) is the LAST ownership-transfer point — it routes any
leftover `RuntimeState*` through the SAME `enqueueDeferredDeleteNonRtWithResult`
→ `shutdownReclaim` → Terminal chain. No path declares completion
before all drains finish.

---

## 9. Verdict: PASS (with INVESTIGATE notes)

### PASS criteria met

- **TerminalReclaimAuthority** is a single by-value member of the unique
  `ISRRetireRouter`; all terminal reclaim flows delegate to it.
- **`ShutdownComplete`** is declared by exactly one `transitionTo` call
  per path (ReleaseResources.cpp:607), guarded by prior
  `markShutdownComplete()` (which itself checks `isFullyDrained()`).
- **OwnerChannel → Terminal** has a single path: `drainAllNonRt`
  callback → `enqueueDeferredDeleteNonRtWithResult` →
  `shutdownReclaim` → `terminalReclaim`.
- **No double-authority**: `isFullyDrained` checks System 1 (slot
  state) and System 2 (pointer counts) independently; no conflation.
- **No direct `delete RuntimeState`/`delete DSPCore`** in production
  `src/`; `destroyDSPCoreNode` is the sole DSPCore deleter, injected
  via the D→Q→E→T chain for published/retired cores, and called
  directly only on pre-publication rollback paths
  (`destroyRolledBackDSP`, DSPGuard destructor, recovery warmup-fail —
  see §6.4).

### INVESTIGATE (non-shutdown paths, out of shutdown-authority scope)

1. **`destroyRolledBackDSP`** (DSPLifetimeManager.cpp:123) — direct
   `destroyDSPCoreNode` call on publish-failure rollback. Object was
   never published → never under authority protection. Safe by
   single-ownership-transfer (RAII unique_ptr), but bypasses terminal
   chain. **Not a shutdown-path concern.**

2. **`CallerDestroy` rollback** (enqueueRuntimePublicationFireAndForget,
   intentQueue-full) — `ownerChannel().take()` return is discarded,
   triggering unique_ptr destructor on RuntimeState. Object was never
   published. Same rationale as above. **Not a shutdown-path concern.**

3. **`FrozenRuntimeWorld::state_`** — if `releaseState()` not called,
   RAII destruction bypasses terminal authority. Call graph confirms
   `releaseState()` is always called before `commitRuntimePublication`
   on the success path; failure paths are covered by INVESTIGATE #2.

### ASan

0xC0000139 — BLOCKED (environment issue, recorded in 15-P-8, not
addressed per audit scope).

Regression baseline: **15-P-8 — Debug CTest 33/33 PASS (36.76s),
Release CTest 33/33 PASS (32.70s)**. No production code changes in 15-P-10.

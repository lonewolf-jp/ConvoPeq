# Phase I-T1-D101-2.5O-3D — Failure/Shutdown/Rejected-Path Ownership Conservation

**Status**: COMPLETE (audit-only, 0 code changes).

## 0. Scope & Prohibitions (per user instruction)

Audit only — zero of:
コード変更 0 / evaluate変更 0 / evaluateDeferred変更 0 / 新 token 0 / 新 permit 0 /
新 ID 0 / 新 deadline 0 / 新 timeout 0 / 新 field 0 / 新 binding API 0 / 設計案の採用 0.

3D inherits 3B/3C findings as facts. 3D = B4 (Terminality disposition) + failure/shutdown path
ownership conservation. 3D does **not** propose hardening (registry asymmetry deferred to 3E if needed).

## 1. Terminal Disposition Map (B4 — 3A §B4 正式判定)

### 1-1 OwnershipDisposition enum (work70 Phase2)

`AudioEngine.h:3581-3594`:

```cpp
enum class OwnershipDisposition : uint8_t {
    None,
    Transferred,
    CallerDestroy
};
struct PublishCommitResult {
    convo::PublishStageResult stage;
    OwnershipDisposition ownership = OwnershipDisposition::None;
};
```

- `Transferred`: publish succeeded, ownership moved to E (ISR). Caller releases claim.
- `CallerDestroy`: publish failed, DSP already rolled back; caller must release via
  `DSPLifetimeManager::destroyRolledBackDSP()`.

- `None`: no ownership transfer occurred (initial/edge).

### 1-2 Acceptance → terminal disposition lattice

All paths from `trySubmitImpl` converge to one of three `OwnershipDisposition` outcomes. The
**O Accepted obligation's world** never vanishes — it is either **Transferred** to E or **CallerDestroy'd**
back to the caller's lifetime manager. No 4th outcome exists (no orphan, no double-destroy).

| Branch | Trigger | OwnershipDisposition | O world fate | E execution? |
| --- | --- | --- | --- | --- |
| Main sync | enqueue OK, intent enqueue OK, executePublish commits | `Transferred` | moves to ISR OwnerChannel → `publish` → store | YES (1:1) |
| Enqueue fail | `OwnerChannel::enqueue` returns false (full/duplicate key) | `CallerDestroy` | `unregister(seqId)` — never left producer | NO |
| Intent queue full | `enqueuePublicationIntent` returns false (queue full) | `CallerDestroy` | `take(key)` reclaims + `unregister(seqId)` | NO |
| Executor publish fail | `executor_.publish` != Success | `CallerDestroy` | `destroyRolledBackDSP(newDSPResolved)` | NO |
| Receipt timeout | `commitRuntimePublication` 250ms timeout | `Transferred` (see §5) | already enqueued/transferred | eventual (ISR still commits) |
| Fire-and-forget | `waitForReceipt=false` | `Transferred` | enqueue + intent enqueue OK | eventual (no producer wait) |

## 2. Failure Branch Census (3C-7/8/9 carry-over)

### 2-1 Payload build failure → RejectedNotFinalized

`RuntimePublicationOrchestrator.cpp:178, 243`: builder returns null/stale → `RejectedNotFinalized`.
Ownership: builder never stamped a world for this O, or stamped but it is `destroyRolledBackDSP`'d
(`RuntimePublicationOrchestrator.cpp:274`). The O's reservation (seqId) may be consumed or not; if
consumed, the world is destroyed before enqueue. **No E execution, no orphan.**

### 2-2 Executor reject → PublishFailed / RejectedPublishFailure

`RuntimePublicationOrchestrator.cpp:260-291`:

```cpp
auto result = executor_.publish(engine_, std::move(frozen), req.newDSP, oldHandle);
if (result != PublishResult::Success) {
    ...
    if (newDSPResolved != nullptr)
        lifetime_.destroyRolledBackDSP(newDSPResolved);
    ...
    if (engine_.isShutdownInProgress())
        return PublicationAdmission::Decision::RejectedShutdown;
    return PublicationAdmission::Decision::RejectedPublishFailure;
}
```

- `destroyRolledBackDSP(newDSPResolved)` (DSPLifetimeManager.cpp:119) → `AudioEngine::destroyDSPCoreNode(dsp)` +
  `fetchAddAtomic(currentRetiringGeneration_, 1)`.

- DSPCore is destroyed; O world object is reclaimed by the producer's lifetime manager.
- `isShutdownInProgress()` race: admission pre-checks shutdown, but publish may fail during a
  mid-flight shutdown. Classified as `RejectedShutdown` (shutdown context) vs `RejectedPublishFailure`
  (internal error). **Ownership disposition is identical** (CallerDestroy) regardless of classification.

### 2-3 Shutdown race → RejectedShutdown

Same locus as 2-2 (RuntimePublicationOrchestrator.cpp:286-290). The shutdown classification is a
**telemetry/diagnostic label** — it does not alter ownership flow. `destroyRolledBackDSP` runs before
the return in both cases.

### 2-4 Admission rejection (pre-Accepted)

`PublicationAdmission.cpp:12, 22` and `RuntimePublicationOrchestrator.cpp:346-367`:
`RejectedShutdown`, `RejectedNotFinalized`, `RejectedPressure`, `RejectedStaleGeneration`,
`RejectedLowPriority`, `RejectedPublishFailure` — all returned **before** `executor_.publish`, so
no world was enqueued. O was never Accepted. **Not in scope** (O Accepted is the entry to 3C/3D).

## 3. Ownership Conservation Theorem (3D core claim)

> **Claim**: Every O Accepted obligation's world reaches exactly one of {Transferred to E,
> CallerDestroy'd by O-side}. There is no path where the world is (a) neither transferred nor
> destroyed (orphan/leak), or (b) both transferred and destroyed (double-free).

### 3-1 Transferred path (happy + timeout + fire-and-forget)

`enqueueRuntimePublicationFireAndForget` / `commitRuntimePublication` (AudioEngine.h:4550-4590):

```cpp
// enqueue OwnerChannel: ownership → E
if (!worldAuthority_.ownerChannel().enqueue(
        convo::isr::OwnerChannelKey{ seqId, epoch, mappedGen }, std::move(world)))
{
    worldAuthority_.registry().unregister(seqId);
    return { convo::PublishStageResult::Failed, OwnershipDisposition::CallerDestroy };
}
// enqueue Intent: ownership instruction → ISR CoordinatorLoop
if (!runtimePublicationBridge_.enqueuePublicationIntent(intent))
{
    (void)worldAuthority_.ownerChannel().take(
        convo::isr::OwnerChannelKey{ seqId, epoch, mappedGen });
    worldAuthority_.registry().unregister(seqId);
    return { convo::PublishStageResult::Failed, OwnershipDisposition::CallerDestroy };
}
// fire-and-forget OR commit (wait for receipt)
rollbackHandle = convo::isr::DSPHandle::null();
return { convo::PublishStageResult::Success, OwnershipDisposition::Transferred };
```

**Key invariant**: the world (unique_ptr of RuntimeState) is `std::move`d into `enqueue`. If enqueue
fails, the unique_ptr was **not** consumed (move is only finalized on success) — but the code calls
`take(key)` to reclaim in the intent-failure branch. The `world` reference is held by the producer's
`stateOwner` (Release.cpp:385-386 `worldOwner.release()` → `frozen`) until `std::move` into enqueue.

On success: `Transferred`. On `commitRuntimePublication` receipt-timeout (250ms, AudioEngine.h:4562),
the disposition is **still Transferred** — the world is already in the OwnerChannel and will be
`take`-n by executePublish on the ISR thread. Timeout is a producer-side wait abandon, not an
ownership event.

### 3-2 CallerDestroy path (failure)

All three failure branches return `{Failed, CallerDestroy}`:

1. `enqueue` fail (AudioEngine.h:4563): `unregister(seqId)` — world never left producer scope
   (move not consumed); `stateOwner`/`frozen` unique_ptr destruction handles the world.
2. Intent enqueue fail (AudioEngine.h:4583): `take(key)` reclaims the moved-into-channel world,
   `unregister(seqId)`, then `stateOwner`/`frozen` destruction.
3. Executor publish fail (RuntimePublicationOrchestrator.cpp:260-291): `destroyRolledBackDSP`
   via `DSPLifetimeManager::destroyDSPCoreNode` + generation retire.

In all cases, exactly **one** reclamation event occurs. The unique_ptr (cases 1, 2) or
`destroyRolledBackDSP` (case 3) is the sole destruction site. **No double-destroy**: enqueue's
`std::move` only consumes the unique_ptr on success (returns true); on failure the unique_ptr
remains valid until scope end.

### 3-3 Double-destroy / orphan analysis

- **Orphan (neither)**: Impossible — the `if (!enqueue) { unregister; return CallerDestroy }` path
  ensures the world is either enqueue'd (Transferred) or explicitly returns CallerDestroy. The
  unique_ptr `world`/`stateOwner` is destroyed at function scope exit for CallerDestroy paths.

- **Double-destroy**: Impossible — `(a)` enqueue succeeds ⇒ `std::move` consumed ⇒ the producer's
  unique_ptr is empty; ISR `take` is sole consumer (take-once proven in 3B-4). `(b)` enqueue fails
  ⇒ producer's unique_ptr intact ⇒ scope-end destruction; `take(key)` in intent-fail branch reclaims
  the channel's copy (which is the producer's, not yet moved). The OwnerChannel `take` CAS-drains
  exactly once (OwnerChannel.h:70-72), so a second take returns null.

- **Registry double-unregister**: `unregister(seqId)` is called in the enqueue-fail and intent-fail
  branches. But these are **mutually exclusive** (`if/else`-adjacent with early return). enqueue-fail
  returns before reaching intent-enqueue. So `unregister` runs at most once per seqId per attempt.
  And `unregister` zeroes the entry (ISRRuntimeWorldAuthority.h:62-66), making it idempotent.

## 4. Shutdown Interaction (3A §B4 "shutdown interaction")

### 4-1 Admission-level shutdown rejection

`PublicationAdmission.cpp:12`:

```cpp
if (engine_.isShutdownInProgress()) {
    return Decision::RejectedShutdown;
}
```

This is a **pre-Accepted** rejection — no world built, no enqueue. O is rejected before Accepted.
**Not in 3D-1 world-conservation scope** (Accepted is the entry).

### 4-2 Publish-level shutdown race

The admission→publish race (`RuntimePublicationOrchestrator.cpp:280-290`): admission passes
(non-shutdown), but publish fails because shutdown began mid-flight. Classified as
`RejectedShutdown`. Ownership: `destroyRolledBackDSP` (CallerDestroy path, §3-2). The world is
reclaimed by the producer's DSP lifetime manager. **No leak**: same disposal as RejectedPublishFailure.

### 4-3 ISR CoordinatorLoop shutdown

When the ISR CoordinatorLoop receives a shutdown signal:

- In-flight intents in the queue are drained or cancelled per shutdown policy.
- A `take(key)` that already succeeded (world transferred to executePublish) — executePublish
  proceeds to `authority.publish` → `publishAndSwap`. If shutdown preempts the swap, the world
  is either published (swap completes) or remains in the OwnerChannel and is reclaimed by
  `drainAllNonRt` at shutdown (OwnerChannel.h: drainAllNonRt idempotent shutdown path).

- A `take(key)` that has NOT yet succeeded (world still in channel) — shutdown drains via
  `drainAllNonRt`, returning the world to the authority for destruction.

**Conservation under shutdown**: the world is either (a) swapped into the store (Published,
lifecycle managed by retire), or (b) drained back to the authority and destroyed. No orphan path
exists because `drainAllNonRt` is idempotent and covers the channel on shutdown.

### 4-4 Deferred lane (control-lane) — NOT O

DeferredPublishSlot is a control-lane pre-admission state (3A §Deferred との混同禁止). O accepted
obligations are in the main lane; Deferred is a separate domain. **No cross-conservation required**
(3A §Deferred との混同禁止の再確認 — O-1/O-2 確定の再確認).

## 5. Receipt Timeout Disposition (3B-6, 3C-8 reaffirmed)

`commitRuntimePublication` (AudioEngine.h:4562-4565):

```cpp
auto committed = waitForPublishReceipt(seqId, 250ms);
if (!committed) {
    // diag log + return Transferred (not failure)
    return { PublishStageResult::Published, OwnershipDisposition::Transferred };
}
```

- **Timeout returns Transferred, NOT CallerDestroy.** The world was already enqueued (ownership
  moved to ISR). The producer's wait abandoning does not reclaim ownership — executePublish on the
  ISR thread will `take(key)` and commit asynchronously.

- This is the **only** case where the producer returns early but ownership is already Transferred.
- 3B-6 established: `OwnershipDisposition::Transferred` encoded at enqueue; timeout preserves
  Transferred; no rollback/double-take path exists.

## 6. Fire-and-Forget Ownership (3A §trySubmit return semantics)

`enqueueRuntimePublicationFireAndForget` (AudioEngine.h:4550): the world is enqueued + intent
enqueued, then `rollbackHandle = null()` and returns `{Success, Transferred}`. The producer
immediately drops its wait obligation. Ownership is identical to the sync path — the world is in
the ISR channel. The producer's `frozen`/`stateOwner` unique_ptr is empty (moved into enqueue).
**No separate ownership state** — fire-and-forget is a wait-suspension, not an ownership change.

## 7. B4 — Terminality — Formal Judgment

> Does every O Accepted obligation reach a terminal disposition, and is ownership conserved
> (no leak, no double-destroy) at every failure/shutdown terminal point?

**Answer — PASS.**

The three terminal ownership outcomes are exhaustive and mutually exclusive:

1. **Transferred** (happy path, receipt timeout, fire-and-forget): world enters OwnerChannel;
   ISR `executePublish` is the sole `take` consumer (take-once, 3B-4). OwnerChannel +
   `drainAllNonRt` (shutdown) covers the world's full lifecycle.
2. **CallerDestroy** (enqueue fail, intent queue full, executor publish fail, shutdown race):
   exactly one reclamation — `unregister` + unique_ptr scope-destruct (enqueue/intent-fail branches)
   or `destroyRolledBackDSP` (publish-fail branch). Mutually exclusive branches (early returns).
3. **None**: only the pre-init state of `PublishCommitResult`; never a terminal disposition
   returned to a caller that carried an Accepted O.

**Leak proof**: every Accepted path returns `{Failed, CallerDestroy}` or `{Success, Transferred}`.
CallerDestroy paths either (a) never moved the unique_ptr (enqueue fail — move not consumed on
failure, confirmed by `std::move` semantics into `enqueue` which returns false), or (b) explicitly
reclaim via `take(key)` + `unregister` (intent-fail), or (c) call `destroyRolledBackDSP`
(publish-fail). No path leaves the world allocated without a destruction site.

**Double-free proof**: enqueue's `std::move(world)` is consumed only on `enqueue(...) == true`.
On failure, the unique_ptr is intact and destroyed at scope exit. ISR `take` is single-transfer
(OwnerChannel.h:70-72 CAS-drain). `drainAllNonRt` (OwnerChannel.h: drain idempotent) is the shutdown
backstop. Registry `unregister` is idempotent (zeroes entry). No two sites can reach the same
RuntimeState* for destruction.

**Code evidence**:

- `OwnershipDisposition` enum (AudioEngine.h:3586-3589).
- Enqueue fail → CallerDestroy (AudioEngine.h:4563).
- Intent enqueue fail → take + unregister + CallerDestroy (AudioEngine.h:4576-4587).
- Executor publish fail → destroyRolledBackDSP + RejectedShutdown/RejectedPublishFailure
  (RuntimePublicationOrchestrator.cpp:268-291).

- Receipt timeout → Transferred (AudioEngine.h:4562-4565, 3B-6).
- Fire-and-forget → Transferred (AudioEngine.h:4589-4590).
- Shutdown drain backstop (OwnerChannel.h: drainAllNonRt idempotent).
- DSP destruction (DSPLifetimeManager.cpp:119-123, destroyDSPCoreNode + generation retire).

## 8. Registry Asymmetry Under Failure (3C-8 / 3B-3 carry-forward)

The seqId-only `PendingPublishRegistry::lookup(seqId)` asymmetry (3C-B2-D) is **re-evaluated
under failure paths**:

- **Enqueue fail**: `unregister(seqId)` runs immediately (AudioEngine.h:4564). Registry entry is
  zeroed. No stale entry survives.

- **Intent enqueue fail**: `take(key)` + `unregister(seqId)` (AudioEngine.h:4577-4579). Same
  seqId as registered at t0. The key is composite but registry is seqId-only — however this failure
  is pre-dispatch (before `executePublish`), so no ISR consumer ever calls `lookup` for this seqId.
  The registry is register→unregister in the same synchronous frame. **No observer.**

- **Executor publish fail**: `destroyRolledBackDSP` runs; `unregister` already ran in
  `enqueueRuntimePublicationFireAndForget` (AudioEngine.h:4516 register → 4519 enqueue). If enqueue
  succeeded (Transferred), the registry entry persists until `executePublish` completes or shuts
  down. If `executePublish` never runs (e.g., shutdown drains the channel without dispatching),
  `drainAllNonRt` reclaims the world but does NOT call `unregister` — **the registry entry goes stale**.

**This is the failure-path gap for B2-D**: if shutdown drains the OwnerChannel without running
`executePublish` (because the ISR loop is torn down), the `PendingPublishRegistry` entry for that
seqId is never `unregister`'d. The entry points to a destroyed world (the `void*` newWorld is
dangling). A subsequent `lookup(seqId)` — only possible if seqId is reused — would return a
dangling pointer.

**Why this is safe today**: seqId is monotonic and never reused within a live session (3B-2, 3C-B2-B).
A crashed-and-restarted session does not reuse seqIds (new `RuntimeBuilder` session starts at
generation boundary). The registry is zero-initialized (`Entry{}` default, ISRRuntimeWorldAuthority.h:
slot zeroed). Stale entries are overwritten, not read, on seqId reuse — and reuse never occurs in a
single session. `drainAllNonRt` operates at shutdown when no new publish can race.

**Verdict on failure-path registry gap**: PASS under the current single-session model. The stale-entry
hazard is structurally contained by seqId-monotonicity + single-loop + idempotent drain. This is the
**same** deferred hardening note from 3C-B2-D (epoch/mappedGeneration on registry Entry) — would only
be needed under a future model that reuses seqIds or allows out-of-order shutdown drain.

## 9. B4 / Failure Conservation — Formal Judgment

| Failure/shutdown branch | OwnershipDisposition | Reclamation site | Leak? | Double-free? |
| --- | --- | --- | --- |
| Main sync (happy) | Transferred | ISR executePublish → publishAndSwap | No | No |
| Enqueue fail | CallerDestroy | unique_ptr scope-destruct + unregister | No | No |
| Intent queue full | CallerDestroy | take(key) reclaim + unregister + scope-destruct | No | No |
| Executor publish fail | CallerDestroy | destroyRolledBackDSP (DSPLifetimeManager.cpp:119) | No | No |
| Receipt timeout | Transferred | ISR executePublish (async) | No | No |
| Fire-and-forget | Transferred | ISR executePublish (async) | No | No |
| Shutdown race (admission-passed) | CallerDestroy | destroyRolledBackDSP | No | No |
| Shutdown drain (channel) | N/A (no Accepted O) | drainAllNonRt idempotent | No | No |
| Admission reject (pre-Accepted) | N/A | N/A (not Accepted) | No | No |

**PASS** — every O Accepted obligation reaches exactly one terminal disposition (Transferred or
CallerDestroy). No leak, no double-destroy path exists in the current model.

## 10. 3D → 3E Transition

3D established:

1. **Terminal disposition lattice**: {Transferred, CallerDestroy} exhaustive, mutually exclusive.
2. **Ownership conservation**: world is either moved to E (1:1 via take-once) or reclaimed by O-side
   (unique_ptr destroy / destroyRolledBackDSP). No orphan, no double-free.
3. **Failure-path gap**: registry stale-entry on shutdown-drain-without-executePublish — structurally
   contained by seqId monotonicity, single session.

3E (Execution Binding Contract Closure) inherits 3D's terminal map and must:

- Produce the **formal Execution Binding Contract** — a judgment table mapping Accepted O disposition
  to terminal E state across all branches (including the fire-and-forget / receipt-timeout async gap).

- Resolve the 3D-8 registry-stale-entry gap for 3E closure (or document it as an accepted
  PROVISIONAL constraint).

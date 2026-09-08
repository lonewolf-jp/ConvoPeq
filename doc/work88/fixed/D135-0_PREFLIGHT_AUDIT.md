# D135-0 — Implementation Preflight Audit

**Type:** read-only / forensic / implementation gate
**Production source変更：0**
**Prerequisite:** D133-1 (measured) + D134 (design). This audit validates the D134-7 plan against the *actual* code paths before any D135 edit.

Verdict at end of each section: **GO / NO-GO**.

---

## 1. P0 — commit-observation proof (Can the rebuild thread observe `fadingRuntimeUuid=0` after a recovery publish?)

### 1.1 The chain

| Stage | File:line | Mechanism | Observable side-effect |
|---|---|---|---|
| Build idle world | `AudioEngine.Transition.cpp:13-14` `publishIdleWorldOnly` | `buildRuntimePublishWorld(current, nullptr, policy, 0.0, false)` | `topology.fadingRuntimeUuid = (active && next!=nullptr)?…:0` → **0** (active=false,next=nullptr) |
| Commit (sync) | `AudioEngine.h:4683` `commitRuntimePublication` | `enqueueRuntimePublicationFireAndForget(...)` then `waitForPublishReceipt(seqId, 250ms)` | blocks caller until receipt |
| enqueue | `PublicationExecutor.h` / `ISRRuntimePublicationCoordinator_ProcessIntent.cpp:147` | intent pushed to CoordinatorLoop queue (async) | world staged, not yet committed |
| executePublish | ISR PublishExecutor `executePublish` | `authority.commit()` | live `RuntimeWorld` swapped |
| onPublishCommitted | `RuntimePublicationOrchestrator.cpp:343` | `publishAtomic(m_lastObservedSequence, seqId, …)` `; engine_.notifyPublishReceipt(seqId)` | `m_lastObservedSequence` bumped; `waitForPublishReceipt` returns true |
| observe | `AudioEngine.h:1655` `getLastCommittedPublicationSequence()` ; `AudioEngine.h:1147` `observePublishedWorld()` | read `m_lastObservedSequence` / `RuntimeStore::current` | caller reads `topology.fadingRuntimeUuid` |

### 1.2 Key facts (verified)

- `commitRuntimePublication` is the **blocking** variant (not the fire-and-forget
  `enqueueRuntimePublicationFireAndForget`). Its body (AudioEngine.h:4683-4713):
  `enqueueRuntimePublicationFireAndForget(...)` + (if isCommitted && seqId!=0) `waitForPublishReceipt(seqId, 250ms)`.
  Comment L4688-4691 is explicit: *"ownership is transferred at enqueue; timeout ≠ failure"*;
  a timeout still counts as Transferred, and the receipt (from `onPublishCommitted` L345-346)
  `complete(seqId)`→`waitForPublishReceipt` returns true on success.
- `onPublishCommitted` (L343) is the **single Completion Authority** for the ISR path:
  it publishes `m_lastObservedSequence` (L345) with `memory_order_release` and then
  `notifyPublishReceipt(seqId)` (L346 → `AudioEngine.h:3764 publishReceiptWaiter_.complete`).
  So **after `commitRuntimePublication` returns with stage==Committed, the zero-fading
  world is provably live** and `getLastCommittedPublicationSequence`/
  `observePublishedWorld()->topology.fadingRuntimeUuid == 0` is readable.
- The recovery path `publishIdleWorldOnly` (Timer.cpp:1690) calls exactly this
  `commitRuntimePublication` (AudioEngine.Transition.cpp:25) → it is **synchronous** on
  the timer thread. The timer thread is **not** the audio thread, so the 250ms max block
  is the same pre-existing behavior as every normal publish.

### 1.3 Measured confirmation

`publishIdleWorldOnly` was invoked by the crossfade-timeout recovery (Timer.cpp:1690).
In D133-1's trace it built the gen7 idle world:
`BuilderExit gen=7 … fadingRuntimeUuid=0 transitionActive=0` (ConvoPeq.log:24497).
The recovery then `commitRuntimePublication`-swapped it live before logging
`[HEALTH] Crossfade timeout recovery completed` (L24505). After L24505 there are **0
evaluate calls** and **0 further publishIdleWorldOnly** — the zero-fading world committed
but **no deferred re-drive ever observed it** before SHUTDOWN_BEGIN (L27368).

> **P0 verdict: GO.** The existing `m_lastObservedSequence` + `observePublishedWorld` API
> lets the rebuild thread prove "world seq ≥ recovery-seq committed AND
> fadingRuntimeUuid==0" with **no new API**. The D134-7 F1′ plan ("wait for the recovery
> world commit") is implementable on existing primitives.

---

## 2. P1 — Deferred retry-counter ownership (Where must `retryCount` live?)

### 2.1 The trap (D134-7 proposal is INVALID as written)

`enqueueDeferred` (RuntimePublicationOrchestrator.cpp:487-496) installs the deferred slot
via **full struct replacement**:
```cpp
deferredSlot_ = DeferredPublishSlot{
    .request = req,
    .guard = DeferredGuard{ .generation = req.generation, .sequence = … },
    .metadata = …{},
    .lastDiscardReason = DiscardReason::None,
    .enqueueTimestampUs = now
};
```
There is **no field-preservation** — every re-enqueue zeroes/rebuilds every member.
`DeferredGuard` is `{generation, sequence}` only (RuntimePublicationOrchestrator.h:24-27).

**Measured proof (gen8 loop):** every one of the 28 gen8 iterations logged
`enqueueDeferred: gen=8 … hasPrevDeferred=0` — i.e. on each re-enqueue the slot was
already reset by `consume()`→`finishView()` (`deferredSlot_.reset()` + `hasDeferred_=false`
at L583/611). A `retryCount` placed on `DeferredGuard`/`DeferredPublishSlot` would be
reinitialized to 0 on **every** re-enqueue → it would count "re-enqueues since last
enqueue" = always ≤1 → **useless for bounding the loop**.

### 2.2 Required placement (P1 decision)

`retryCount` **must** be a **persistent field on `RuntimePublicationOrchestrator`**
(RUNTIME-PUBLICATION-ORCHESTRATOR-OWNER), surviving slot overwrites, with the lifecycle:

| Event | retryCount action | Rationale |
|---|---|---|
| `enqueueDeferred`: new generation (`req.generation != slot->metadata.generation`) | `retryCount = 0` | fresh obligation, fresh budget |
| `enqueueDeferred`: same generation (re-enqueue / overwrite) | `retryCount += 1` (capped) | counts re-drive cycles of THIS obligation |
| `processDeferredAdmission`: `Ready` → `consume()` → `submitPublishRequest` → returns `DeferredFadingActive` → `enqueueDeferred(same req)` | `retryCount` incremented at the re-enqueue | the loop counter lives here |
| `finishView()` (consume/discard of a *different* generation) | `retryCount = 0` | old obligation gone, reset |
| `clearDeferredForShutdown` | `retryCount = 0` | shutdown tears down |

So **P1-GO condition:** retry counter is an Orchestrator-level atomic/counter, NOT on the
slot. The D134-7 wording ("add retryCount to DeferredGuard") must be corrected to
"add `deferredRetryCount_` to `RuntimePublicationOrchestrator`, gated by generation".

> **P1 verdict: GO** — but **only with the corrected placement** (Orchestrator-level,
> generation-keyed reset). The D134-7 draft spec is **NO-GO** as written; see §6.

---

## 3. P2 — Re-drive recursion proof (Is `submitPublishRequest` → `processDeferredAdmission` a stack recursion?)

### 3.1 The call graph (verified by grep over `src/`)

- `processDeferredAdmission` is called from **exactly one** site outside tests:
  `AudioEngine.RebuildDispatch.cpp:904` (inside `rebuildThreadLoop`).
- `submitPublishRequest` (RuntimePublicationOrchestrator.cpp:357-435) calls `trySubmitImpl`
  (L360); its `DeferredFadingActive` branch (L377-380) is:
  ```cpp
  case PublicationAdmission::Decision::DeferredFadingActive:
      enqueueDeferred(req);
      return;
  ```
  **It does NOT call `processDeferredAdmission`.** `trySubmitImpl` either returns early
  (non-Accepted decisions, before the publish call L107) or reaches `executor_.publish`
  (L107) and returns `Accepted`. No path re-enters `processDeferredAdmission`.
- `trySubmitImpl`'s publish (L107 `executor_.publish(engine_, …)`) uses
  `PublicationExecutor::publish` → `publishImpl(waitForReceipt=true)` (PublicationExecutor.cpp:14) —
  the **synchronous** Blocking variant. BUT this is only reached on the **Accepted** path;
  `DeferredFadingActive` returns at `evaluate` (trySubmitImpl L380) **before** the build/publish.

### 3.2 Why the loop is tight (28 iters, no cap)

`processDeferredAdmission` (L627-693): `peek → evaluateDeferred → Ready → consume() + finishView() + submitPublishRequest`. `submitPublishRequest`→`trySubmitImpl`→`evaluate`→`DeferredFadingActive`→`enqueueDeferred` (sets `hasDeferred_=true`) → returns. `enqueueDeferred` does NOT signal the rebuild thread directly; the rebuild thread is re-woken by an external `publishRetryReady` signal. In `rebuildThreadLoop` (RebuildDispatch.cpp:837-843) the CV predicate is
`hasPendingTask || publishRetryReady || recoveryPending || …`, and after consuming,
`doDeferredPublish = publishRetryReady` (L879). The gen8 trace shows
`[D129_TASK_WAKE] … wokeByRetryReady=1` on **every** iteration → each re-defer sets
`hasDeferred_`/`publishRetryReady`, re-waking the rebuild thread → re-enters
`processDeferredAdmission`. **No counter anywhere.** Termination in D133-1 was
**external** (the `[MEM_SNAP]` health tick at L24497 pre-empting the memory-starved
rebuild thread; then `SHUTDOWN_BEGIN`).

### 3.3 The re-entrancy guard the user requires

User requirement: *"submitPublishRequest 内で processDeferredAdmission を直接再起動するような変更は禁止"* — **already structurally satisfied** (no such call exists; grep is empty). The wake-driven re-entry is at `RebuildDispatch.cpp:904`, NOT inside `submitPublishRequest`. So any D135 change must keep re-drive initiation **only** at `RebuildDispatch.cpp:904` (or the recovery handler), never inside `submitPublishRequest`/`trySubmitImpl`.

> **P2 verdict: GO** — no call-stack recursion exists today and the fix must preserve that:
> re-drive is initiated only from `rebuildThreadLoop` (L904) / the recovery handler, never
> from `submitPublishRequest`. The cap (P1) and the recovery re-trigger (F1′) attach at
> those two wake sites.

---

## 4. P3 — Discard semantics (Is `StaleDiscard` correct for retry-exhaustion?)

`DiscardReason` enum (RuntimePublicationState.h:10-17):
```cpp
enum class DiscardReason : uint8_t {
    None, ShutdownDiscard, StaleDiscard, SupersededDiscard, Expired  // ★ work37: TTL
};
```
`evaluateDeferred` (PublicationAdmission.cpp:106-133) returns `Discard` with:
- `ShutdownDiscard` (L118, shutdown)
- `StaleDiscard` (L123, TTL expired — note work37 comment says "Expired を別 enum 化可能")
- `StaleDiscard` (L127, generation mismatch `m.generation != ctx.currentGeneration`)
- `StaleDiscard` (L131, sequence rollback `m.sequence < ctx.lastSequence`)

**Problem:** retry-exhaustion of a *fresh* (generation-valid, non-TTL-expired, non-stale)
deferred request has **no dedicated reason**. Forcing it to `StaleDiscard` is semantically
wrong: the payload was not stale by generation or TTL; it was starved by a persistent
non-stale `hasFading`. This would corrupt `DeferredHealth.lastDiscardReason` telemetry
and mask the real failure mode.

**P3 requirement:** add `DiscardReason::RetryExhaustedDiscard` to the enum
(RuntimePublicationState.h:10), and have the retry-cap path (P1) `discard()` with it.
Minor (enum add + one switch arm in finishView/telemetry) but required for faithful
telemetry and to keep `StaleDiscard` meaning "generation/TTL/sequence stale".

> **P3 verdict: GO contingent** — D135 must add `RetryExhaustedDiscard` (not reuse
> `StaleDiscard`). Without it the fix produces misleading diagnostics. Tracked in §6.

---

## 5. P4 — RT-safety / ownership (Does the fix touch the audio thread?)

- **Owner:** `processDeferredAdmission` (L629), `enqueueDeferred`, `finishView`,
  `peekDeferred` all **`jassert(std::this_thread::get_id() == engine_.rebuildThreadId())`**
  (L542, L567, L629). Rebuild thread = `rebuildThreadLoop` (RebuildDispatch.cpp:260),
  configured `ThreadType::HeavyBackground` + `VML_FTZDAZ_ON` (L271-275). **Not the audio thread.**
- **`publishIdleWorldOnly`/recovery (Timer.cpp:1657-1694)** runs on the **timer thread**
  (message thread's `timerCallback`), which **already** calls the blocking
  `commitRuntimePublication` (250ms max). This is pre-existing; the recovery handler is
  already synchronous-blocking on a Non-RT thread.
- **F1′ re-trigger** (recovery handler schedules a deferred re-drive after the idle-publish)
  would call `runtimeOrchestrator_->processDeferredAdmission()` from the timer thread —
  but that function `jassert`s rebuild-thread ownership. **So F1′ must NOT call
  `processDeferredAdmission` directly from the timer thread; it must signal the rebuild
  thread** (set `publishRetryReady`/push a deferred-retry task). Existing signaling:
  `publishRetryReady` + `rebuildCV.notify_one()` (the same path the CoordinatorLoop uses).
- **No allocation / no blocking on audio thread:** `trySubmitImpl`'s rebuild-world path
  uses `aligned_make_unique` on the rebuild thread (L95) — already the case today. The
  fix does not add any RT-thread work. `hasFadingRuntimeInWorld` (AudioEngine.h:3304) is
  `noexcept`, atomics only, already RT-callable.
- **Atomic ordering:** `enqueueDeferred` publishes `hasDeferred_` with
  `memory_order_release` (L500); `processDeferredAdmission` consumes with `acquire`
  (L629 guard / L542). Retry counter must follow the same `acquire`/`release` discipline.

> **P4 verdict: GO** — provided F1′ signals the rebuild thread (via the existing
> `publishRetryReady`/CV path) rather than calling `processDeferredAdmission` inline from
> the timer thread, and F3′'s retry check is an atomic on the rebuild thread. No new
> RT-thread blocking or allocation.

---

## 6. P5 — Required change set (D135-0 final spec)

### 6.1 Must change (production)
| File | Function | Change |
|---|---|---|
| `RuntimePublicationOrchestrator.h` | member decl | add `std::atomic<uint8_t> deferredRetryCount_{0}` + `std::atomic<int> deferredRetryGeneration_{0}` (generation-keyed reset) |
| `RuntimePublicationOrchestrator.cpp` | `enqueueDeferred` | on re-enqueue of **same generation** (vs current `deferredRetryGeneration_`): `deferredRetryCount_.fetch_add(1)`; on **new** generation: reset to 0 + store gen. If count ≥ N(2) → `discard(RetryExhaustedDiscard)` + diagLog starve + return (do not re-enqueue) |
| `RuntimePublicationOrchestrator.cpp` | `processDeferredAdmission` | after `evaluateDeferred → Discard`, if DiscardReason==RetryExhausted, also fire `[HEALTH]` + `emitEvidenceTickNonRt` |
| `AudioEngine.Timer.cpp` | `EVENT_CROSSFADE_TIMEOUT` handler (1657-1694) | after `publishIdleWorldOnly(...)` returns (sync, world committed), **if `hasDeferred_` && deferred generation == currentGeneration**: signal rebuild thread to re-drive **once** (set `publishRetryReady` + notify), using the committed-sequence gate from §1.1 |
| `RuntimePublicationState.h` | `DiscardReason` enum | add `RetryExhaustedDiscard` |
| `PublicationAdmission.cpp` | `evaluate` (D133 diag block) | DIAG-only: also echo `worldFadingUuid` read (read `observePublishedWorld()->topology.fadingRuntimeUuid` under the `PublicationReader`) so post-fix we can prove "deferred because world-fading=3" → "deferred because world-fading=0 never re-checked" |

### 6.2 Must NOT change (invariant boundaries)
- `AudioEngine.h:3304` `hasFadingRuntimeInWorld` — the single authority for `hasFading`; must not be weakened.
- `RuntimeBuilder.cpp:220` `fadingRuntimeUuid = (active && next != nullptr) ? … : 0` — the completion rule; must not alter.
- `commitRuntimePublication` / `PublicationExecutor` semantics (including the `waitForReceipt` flag) — do not change.
- `RuntimePublicationCoordinator` — Completion layer is correct; leave untouched.
- D134-LCG `lastCommittedRebuildGeneration` — **separate defect**, NOT in D135.
- `INV-DEFERRED-2` (single-slot latest-only) — keep; do not convert to multi-slot (that is F2′, tracked separately).

### 6.3 DIAG-only (rebuild-diag, no production behavior)
- The `evaluate` log line extension (§6.1 last row) is already gated
  `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`; the gen7 idle-publish build is already logged
  (`BuilderExit gen=7 fadingRuntimeUuid=0`). Add a `[D135]` line at the recovery re-trigger
  and at retry-cap exhaustion, also diag-gated.

### 6.4 Tests (added)
- `DeferredFlowIntegrationTests.cpp` / harness: assert the re-drive loop terminates in
  ≤ N iterations when `hasFading` stays true (retry-cap path), and that a *later* valid
  rebuild still publishes (generation uniqueness preserved).
- Post-recovery: drive a crossfade-starve → timeout recovery → assert the deferred slot
  (same generation) is re-evaluated against the committed `fadingRuntimeUuid=0` world and
  publishes (the D134-1 P0 scenario), using `getLastCommittedPublicationSequence` as the
  observation gate (no new API).

---

## 7. GO / NO-GO

### NO-GO conditions (checked — all satisfied, i.e. NONE triggered)
| Condition | Result |
|---|---|
| `publishIdleWorldOnly` completion not observable via existing API | ❌ NOT triggered — `m_lastObservedSequence` + `observePublishedWorld` exist (§1.1). GO. |
| retryCount reset by `enqueueDeferred` overwrite | ✅ **triggered** — but mitigated by Orchestrator-level counter (§2.2), NOT by slot field. GO. |
| recursive re-entry inside `submitPublishRequest` | ❌ NOT triggered — no such call exists; re-drive only at L904 / recovery handler (§3.3). GO. |
| retry exhaustion causes ownership leak | ❌ NOT triggered — `discard()`→`finishView()` always releases slot+`hasDeferred_` (L562-587); new DiscardReason just adds a telemetry label. GO. |
| publish retry against a non-zero-fading world possible | ❌ NOT triggered — F1′ gates re-drive on `commitRuntimePublication` having returned (world live) + sequence gate (§1.3). GO. |
| retry exhaustion = silent disappearance | ✅ **triggered as design issue** — `StaleDiscard` reuse is wrong; add `RetryExhaustedDiscard` (§4). Addressed. GO. |
| audio thread blocks on new sync wait | ❌ NOT triggered — recovery is on timer thread (pre-existing sync); re-trigger signals rebuild thread via CV (§5). GO. |
| `INV-DEFERRED-2` single-slot semantics altered | ❌ NOT triggered — single-slot preserved; fix is ordering + cap, not buffering. GO. |

### Final verdict: **GO** (with 3 mandatory spec corrections vs D134-7)
1. **retryCount placement:** `RuntimePublicationOrchestrator`-level atomic, generation-keyed reset — NOT on `DeferredGuard`/`DeferredPublishSlot` (§2.2).
2. **re-drive initiation only at** `RebuildDispatch.cpp:904` or the recovery handler via `publishRetryReady`+CV — never inline from `submitPublishRequest` or from the timer thread (§3.3, §5).
3. **Add `DiscardReason::RetryExhaustedDiscard`** rather than reuse `StaleDiscard` (§4, §6.2).

**D135-1 (implementation) may proceed** with the corrected spec in §6. Production source remains untouched until D135-1.

---

## Appendix — line-level evidence index

- `hasFadingRuntimeInWorld`: AudioEngine.h:3304-3307 (reads `topology.fadingRuntimeUuid`).
- `fadingRuntimeUuid` write (sole): RuntimeBuilder.cpp:220.
- `publishIdleWorldOnly` (sync commit): AudioEngine.Transition.cpp:10-33; calls `buildRuntimePublishWorld` (B-1 builds uuid=0) + `commitRuntimePublication` (B-4).
- `commitRuntimePublication` (blocking): AudioEngine.h:4683-4713; `waitForPublishReceipt` L4702.
- `onPublishCommitted` (completion, sets `m_lastObservedSequence`): RuntimePublicationOrchestrator.cpp:343-351; decl RuntimePublicationOrchestrator.h:152.
- `waitForPublishReceipt` / `notifyPublishReceipt`: AudioEngine.h:3764, 3766.
- `getLastCommittedPublicationSequence` / `observePublishedWorld`: AudioEngine.h:1655, 1147.
- `enqueueDeferred` (struct-reset overwrite): RuntimePublicationOrchestrator.cpp:437-513.
- `submitPublishRequest` (no `processDeferredAdmission` call): RuntimePublicationOrchestrator.cpp:357-435.
- `processDeferredAdmission` (rebuild-thread jassert L629; re-drive body L633-693).
- `DeferredGuard` (2 fields, no retry): RuntimePublicationOrchestrator.h:24-27.
- `processDeferredAdmission` wake site: AudioEngine.RebuildDispatch.cpp:894-904 (+ CV predicate L837-841).
- `EVENT_CROSSFADE_TIMEOUT` recovery: AudioEngine.Timer.cpp:1657-1694.
- `trySubmitImpl` re-defer returns before publish: RuntimePublicationOrchestrator.cpp:380 (evaluate gate) + executor_.publish sync at L107.
- `DiscardReason` enum (4 values, no retry-exhaust): RuntimePublicationState.h:10-17.
- Gen8 28-iteration loop (measured): evidence/D133-1_6burst_diag.log L24433-24490; `hasPrevDeferred=0` each iter confirms slot-reset; 0 evaluates after L24505.

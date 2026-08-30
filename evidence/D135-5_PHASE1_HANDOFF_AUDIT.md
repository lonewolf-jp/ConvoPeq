# D135-5 — Phase 1 Handoff / Recovery Signal Provenance Audit

**Status:** Phase-1 Design Audit (read-only, no source changes)
**Predecessor:** `evidence/D135-4_DESIGN_AUDIT.md` (Phase 0, GO to Phase 1)
**Scope:** Verify that the **wake provenance** ("ordinary Deferred retry" vs "Recovery timeout") can be distinguished at the rebuild-thread consumer before wiring Phase 1's `recoveryArmed=true` + `resetDeferredRetryBudget` plan.
**Conclusion:** **NOT distinguishable.** The D135-4 Phase-1 plan is **blocked** until a dedicated recovery-wake signal is introduced. This audit confirms the latent discriminator (`lastRecoveryPublishSeq_`) is **write-only / never consumed**, and that the retry-budget reset must NOT fire on ordinary retries — which the blind consumer cannot enforce.

---

## 1. Executive Summary

D135-4 locked the **thread-ownership** boundary for the deferred-publish retry members (plain `deferredRetryGeneration_` / `deferredRetryCount_` must only be touched by the rebuild thread; the MessageThread write site at `resetDeferredRetryBudget()` is the race). D135-5 extends this to the **wake-provenance** boundary:

1. `publishRetryReady` is set `true` by **two distinct producers** — an *ordinary* Coordinator retry (`runCoordinatorPhase`) and the *recovery* crossfade-timeout handler (`timerCallback`) — then merged into a single `doDeferredPublish` consumption point.
2. The rebuild-thread consumer (`processDeferredAdmission()`) receives **no provenance argument** and **never reads** `recoveryPublishSeq()` / `lastRecoveryPublishSeq_`.
3. The only recovery-tagging signal (`lastRecoveryPublishSeq_`, stamped via `setRecoveryPublishSeq`) is **write-only** — its reader `recoveryPublishSeq()` has **zero callers** in `src/`.
4. Therefore a Phase-1 rule "consume `publishRetryReady` → stamp `recoveryArmed=true` + reset retry budget" would **reset the budget on every ordinary retry**, collapsing the `kMaxDeferredRetries=2` exhaustion bound (C1/C2/C3) and **re-introducing the very loop the counter is meant to bound**.

**Gate verdict:** Recovery/ordinary wake distinction is **not currently identifiable**. Phase 1 source edits remain **blocked**.

---

## 2. `publishRetryReady` — Exhaustive Producer Inventory

`bool publishRetryReady` — declared `AudioEngine.h:2718`, guarded by `rebuildMutex` (acquired only in `AudioEngine.h:2699` / `AudioEngine.RebuildDispatch.cpp`). Single consumer at `RebuildDispatch.cpp:879-880`.

### 2.1 All `publishRetryReady = true` write sites

| # | File:Line | Enclosing function | Thread | Meaning (semantics) |
|---|-----------|--------------------|--------|---------------------|
| P1 | `AudioEngine.Threading.cpp:286` | `AudioEngine::runCoordinatorPhase()` | CoordinatorLoop thread | **Ordinary Deferred retry.** Sets flag only when `hasDeferredRequest()` is true, purely to request the rebuild thread re-evaluate a stuck deferred publish. **Does not** stamp a recovery seq; **does not** reset retry budget. |
| P2 | `AudioEngine.Timer.cpp:1744` | `AudioEngine::timerCallback()` — `EVENT_CROSSFADE_TIMEOUT` handler (§3) | MessageThread (100 ms Non-RT sampler, `Timer.cpp:432/475`) | **Recovery timeout wake.** Follows `setRecoveryPublishSeq(recoverySeq)` (cpp:1723), `resetDeferredRetryBudget()` (cpp:1724), and a HardReset idle-world commit. Intended to give the obligation a *fresh* retry budget. |

No other `= true` sites. (All other grep hits are the `false` clears at `CtorDtor.cpp:182`, `PrepareToPlay.cpp:82`, `ReleaseResources.cpp:175`, and consumer clears at `RebuildDispatch.cpp:861` (shutdown) and `:880` (consumed).)

### 2.2 Producer semantics are **not** distinguished at the flag

`publishRetryReady` is a single `bool`. Both producers write `true` under `rebuildMutex` and `notify_one()`. The flag carries **no cause field**. The comment at `RebuildDispatch.cpp:846-847` ("CoordinatorLoop が publishRetryReady を立てた時に起床する") documents only the ordinary-retry provenance — the recovery-wake provenance (P2) is **invisible** to the consumer.

---

## 3. The Recovery Timeout Handler (`AudioEngine.Timer.cpp:1669-1750`)

Reconstructed verbatim from Read (this section reproduces the exact logic the Phase-1 plan depends on):

```cpp
// Timer.cpp:1669  (runs in timerCallback() — MessageThread)
if (event.eventCode == convo::EVENT_CROSSFADE_TIMEOUT) {
    // 1. CAS-clear fadingRuntimeDSPSlot, submitObserve(...)           [1674-1687]
    // 2. crossfadeAuthorityRuntime_.getActiveCrossfades() → unregister   [1689-1698]
    // 3. crossfadeRuntime_.complete()                                [1701]
    // 4. publishIdleWorldOnly(..., HardReset)  — commits zero-fading world  [1712-1716]
    // 5. RECOVERY GATE:
    if (runtimeOrchestrator_ != nullptr && runtimeOrchestrator_->hasDeferredRequest()) {
        const auto recoverySeq = getLastCommittedPublicationSequence();
        runtimeOrchestrator_->setRecoveryPublishSeq(recoverySeq);   // [1723] atomic write
        runtimeOrchestrator_->resetDeferredRetryBudget();            // [1724] plain write — RACE
        // DIAG log: hasDeferred=1 hardcoded, recoveryWorldSeq, currentGen   [1733-1741]
        {
            std::lock_guard<std::mutex> lock(rebuildMutex);
            publishRetryReady = true;                                // [1744]
        }
        rebuildCV.notify_one();                                     // [1746]
    }
}
```

**Critical observation:** This handler performs **three** distinct cross-thread actions that must all be *recovery-gated*:
- (a) `setRecoveryPublishSeq(seq)` — stamp the committed idle-world seq (atomic, safe).
- (b) `resetDeferredRetryBudget()` — **plain write to retry members from MessageThread**.
- (c) `publishRetryReady = true` — request rebuild-thread wake.

The Phase-1 design correctly identifies that (b) must move to the rebuild thread. But (b)+（c) are currently *interleaved on the MessageThread*, and the rebuild thread wakes to `publishRetryReady` with **no way to know (a) was set for this wake** — let alone that this wake is specifically the recovery wake rather than the ordinary one.

---

## 4. Consumer Side — `publishRetryReady` is Blind to Provenance

### 4.1 The single consumption point

`AudioEngine.RebuildDispatch.cpp:879-880` (inside the `rebuildMutex` critical section that also resolves the CV predicate `cpp:849-854`):

```cpp
doDeferredPublish = publishRetryReady;   // [879]  ← provenance-lost merge
publishRetryReady = false;               // [880]
```

`doDeferredPublish` is then forwarded — with **no provenance argument** — at `cpp:904`:

```cpp
if (doDeferredPublish && runtimeOrchestrator_ != nullptr)
    runtimeOrchestrator_->processDeferredAdmission();   // [904]  ← no arg carries "why we woke"
```

### 4.2 `processDeferredAdmission()` takes no provenance, reads none

`RuntimePublicationOrchestrator.cpp:636-663` (verbatim):

```cpp
void RuntimePublicationOrchestrator::processDeferredAdmission() noexcept {
    jassert(std::this_thread::get_id() == engine_.rebuildThreadId());   // [638] rebuild-thread
    if (!convo::consumeAtomic(hasDeferred_, std::memory_order_acquire))  // [639]  ← only signal read
        return;
    auto view = peekDeferred();                                         // [642]  ← slot peek
    if (!view.has_value()) return;
    auto result = admission_.evaluateDeferred(view->metadata(),         // [646]  ← 4-level:
                             buildDeferredAdmissionSnapshot());         //              Shutdown→TTL→Gen→Seq
    switch (result.decision) {
        case Ready:    { auto req = view->consume(); view.reset();       // [650-651]
                         submitPublishRequest(req); } break;            // [653] recurse → enqueueDeferred
        case Discard:  { view->discard(result.discardReason);           // [658]
                        view.reset(); } break;
    }
}
```

**Neither `processDeferredAdmission` nor its direct sub-calls (`peekDeferred`, `evaluateDeferred`, `consume`, `discard`, `submitPublishRequest`/`enqueueDeferred`) consult `recoveryPublishSeq()`, `lastRecoveryPublishSeq_`, `recoveryPending`, or any wake-reason field.** The only cross-thread signal read is `hasDeferred_` (atomic bool) — which merely says "a deferred obligation exists", not "why we are re-evaluating it now".

### 4.3 The latent discriminator is write-only

`recoveryPublishSeq()` accessor (`RuntimePublicationOrchestrator.h:167-169`):
```cpp
[[nodiscard]] PublicationSequenceId recoveryPublishSeq() const noexcept {
    return convo::consumeAtomic(lastRecoveryPublishSeq_, std::memory_order_acquire);
}
```

Grep across `src/` for `recoveryPublishSeq\(\)` (the callable form) — excluding the definition — returns **zero call sites**. The member `lastRecoveryPublishSeq_` (h:281) is:
- **written** `release` by `setRecoveryPublishSeq` (Timer.cpp:1723, recovery-wake only),
- **reset to 0** `release` by `clearDeferredForShutdown` (cpp:551),
- **read** by `consumeAtomic` in `recoveryPublishSeq()` — but that function is **never invoked**.

So `lastRecoveryPublishSeq_` is a **latch that is set but never consumed**. It cannot, in its current form, carry per-wake provenance: after one recovery wake stamps `seq=S`, every subsequent ordinary retry (P1) would still observe `recoveryPublishSeq()==S` and be misclassified as recovery. This is the classic *stale-signal* defect.

---

## 5. Retry-Budget Invariant Analysis — Why Blind Consumption Breaks kMax=2

### 5.1 The counter is generation-keyed and incremented at re-defer

`enqueueDeferred` (`RuntimePublicationOrchestrator.cpp:470-478`) increments `deferredRetryCount_` for same-generation re-defers:

```cpp
const bool sameObligation = (req.generation == deferredRetryGeneration_);   // [472]
if (sameObligation)  ++deferredRetryCount_;                                  // [474]
else { deferredRetryGeneration_ = req.generation; deferredRetryCount_ = 0; } // [476-478]
if (deferredRetryCount_ >= kMaxDeferredRetries) → RetryExhaustedDiscard, return; // [489-502]
```

Re-defer happens *only* via `processDeferredAdmission` → `Ready` → `consume` → `submitPublishRequest` → (re-defer) `enqueueDeferred` (`cpp:653 → cpp:378`). So the counter is exercised on **every** successful re-evaluation, regardless of wake provenance.

### 5.2 The required split (D135-4 + D135-5)

| Wake source | `retryCount` action | `retryGeneration` action | Target discard |
|---|---|---|---|
| **Ordinary retry** (P1) | `++count` (toward kMax=2) | keep | `RetryExhaustedDiscard` after count 0→1→2 |
| **Recovery wake** (P2) | `count = 0` (fresh budget) | reset to current build gen | none — obligation gets 2 fresh retries |

### 5.3 The invariant table that Phase 1 must preserve

**Ordinary-only exhaustion (must be reachable):**
```
T0 enqueue            gen=5  count=0
T1 ordinary retry     -> consume -> resubmit -> enqueueDeferred -> count=1
T2 ordinary retry     -> count=2  -> RetryExhaustedDiscard  (count 0→1→2, then give up)
```

**Recovery re-armed (must reset, not increment):**
```
T0 enqueue            gen=5  count=0
T1 ordinary retry     -> count=1
T2 recovery wake       -> count RESET to 0  (fresh 2 retries granted)
T3 ordinary retry     -> count=1
```

### 5.4 Why D135-4's blind "consume→reset" plan violates the invariant

If Phase 1 implements "every time `processDeferredAdmission` runs and re-defers, reset the budget", then **every** wake — including P1 ordinary retries — resets `count` to 0. The D135-2 exhaustion trace (`retry 0→1→2→starved`, `recovery-redrive = 0`) would become **unreachable**: count would snap back to 0 on each resubmit and `RetryExhaustedDiscard` could never fire. The `kMax=2` loop bound is silently disabled.

The recovery reset must be **gated on provenance** — "this wake is a recovery wake" — and provenance is currently unavailable to the rebuild thread.

---

## 6. Broader Race Class — `clearDeferredForShutdown` is Also MessageThread-Touched

D135-4 scoped the race to the single `resetDeferredRetryBudget()` call at `Timer.cpp:1724`. D135-5 finds the race class is **broader**: `clearDeferredForShutdown()` writes the *same* plain members (`deferredRetryGeneration_=0; deferredRetryCount_=0;` at `RuntimePublicationOrchestrator.cpp:548-549`) and is called from **MessageThread** at three sites in `timerCallback`:

| `Timer.cpp` line | Handler context | Member writes (plain, unguarded) |
|---|---|---|
| **1642** | `EVENT_PUBLICATION_STALL` → force-drain deferred publish | cpp:548-549 via `clearDeferredForShutdown` |
| **1804** | `RecoveryAction::Recover` → drain + reclaim + release deferred | cpp:548-549 via `clearDeferredForShutdown` |
| **1824** | `Restore` path → epoch recovery idle publish | cpp:548-549 via `clearDeferredForShutdown` |

These are *semantically intentional full resets* (drain-on-stall / epoch-restore), distinct from the recovery *re-arm*. But they confirm the retry members are written from MessageThread via **at least four** call sites (one direct `resetDeferredRetryBudget` + three `clearDeferredForShutdown`), all outside `rebuildMutex`, racing the rebuild-thread reads at `enqueueDeferred` cpp:472. (`clearDeferredForShutdown` itself carries no rebuild-thread `jassert`, unlike `peekDeferred` cpp:566 / `processDeferredAdmission` cpp:638 / `finishView` cpp:591.)

The ReleaseResources.cpp:359 caller of `clearDeferredForShutdown` occurs in shutdown and is single-owner-safe by liveness; the Timer.cpp callers are the racy ones.

---

## 7. `recoveryArmed` Slot-Lifecycle vs. N3 Overwrite-Invariance

D135-4 §4 proposes `bool recoveryArmed{false}` inside `DeferredPublishSlot` (proposed h:32-37), stamped by the rebuild thread. Analyzed against N3 (overwrite must not leak a stale `true`):

### 7.1 Slot replacement is whole-struct, so a defaulted field is safe at construction

`enqueueDeferred` rebuilds the slot via a single aggregate-init at `cpp:505` (`deferredSlot_.emplace(...)`), and the only other construction site is none (grep: one aggregate-init site). INV-DEFERRED-2 (cpp:460-468) retires the *previous* `newDSP` on overwrite but does **not** reset arbitrary slot fields — however, because the slot is *struct-replaced* (not field-patched), a fresh `recoveryArmed{false}` default on the new emplace automatically clears any prior value. So N3's "no leak on overwrite" holds **structurally** for a defaulted field.

### 7.2 The residual risk is the *stamp timing*, gated by provenance

The danger is not overwrite leakage but **mis-stamp**: if `recoveryArmed=true` is stamped whenever `publishRetryReady` is consumed (the D135-4 §8.2/§8.3 plan), then an ordinary retry that re-defers will stamp `recoveryArmed=true` (because the consumer cannot tell it apart from recovery), and the *next* `processDeferredAdmission` sees `recoveryArmed==true` and resets the budget — violating the ordinary-exhaustion invariant (§5.4). So `recoveryArmed` placement is correct; its **trigger** is the defect. The field must be stamped **only** on a wake proven to be recovery-originated.

---

## 8. Design Options for a Provenance Signal (Phase 1, pre-edit)

This section records candidate mechanisms **without committing** to an edit (per D135-5 §10, no source change at this gate):

| Option | Mechanism | Pros | Cons / open question |
|---|---|---|---|
| **D1** | Dedicated `std::atomic<bool> recoveryRetryReady` (set only by P2 under `rebuildMutex`, cleared by consumer). `processDeferredAdmission` takes a `bool wasRecoveryWake` arg. | Minimal; mirrors existing `publishRetryReady` shape; rebuild-thread owns the reset. | Adds one atomic + one arg. Must verify no third producer is missed. |
| **D2** | Repurpose `recoveryPublishSeq()` as **consume-on-read**: rebuild thread reads it once per admission; if `!= 0` → recovery wake (then `publishAtomic(...,0)` to clear). | Reuses the write-only latch; `seq` already carries world identity for telemetry. | Stale-seq across ordinary retries unless cleared on read; the ordinary Retry path never sets it, so `!=0` ⇒ recovery is sound *if* cleared-on-consume. Needs the rebuild side to call `recoveryPublishSeq()` (currently zero callers). |
| **D3** | Enum `deferredWakeupReason{Ordinary, Recovery, Shutdown}` set by each producer under `rebuildMutex`; single CV predicate field. | Unambiguous; self-documenting. | Slightly larger footprint; touches the CV predicate block. |
| **D4** | Thread-local or per-wake parameter passed into `processDeferredAdmission(bool)`. | No shared state change; explicit. | Only feasible if the wake-decision and the call are co-located, which they are (cpp:898-904). |

**Preference order emerging (pre-edit):** D4 (explicit arg from `doDeferredPublish` site, no new shared atomics) ≈ D1 (dedicated flag) > D2 (consume-on-read repurposing) > D3 (predicate rewrite). The choice is deferred **until provenance is proven distinguishable** — which, today, it is not (§4.3).

---

## 9. Phase 2 / Phase 3 Implication Note (D135-2 Re-run)

The D135-4 audit scheduled a D135-2 harness re-run at `--cli-intent-burst-interval-ms 4000` with the expectation that `recovery-redrive > 0` would be observed post-fix. **This expectation is only valid once a provenance signal exists**: with the blind consumer, the "recovery" re-drive is indistinguishable from ordinary retry and the budget reset cannot be scoped, so the `retryCount 0→1→2→starved` exhaustion (and thus the `starved → recovery-redrive = 0` signature) would either spuriously not occur (if reset fires on every retry) or spuriously recur. The re-run gate therefore inherits the provenance prerequisite.

---

## 10. Corrected Phase-1 Edit Checklist (revised — provenance prerequisite added)

The D135-4 §10.1 5-site checklist is **revised**: edits 1-3 and 5 are unchanged; **edit 4 gains a prerequisite** — a provenance signal must be selected and wired before the budget reset can be relocated.

| # | Edit (D135-4) | Status under D135-5 |
|---|---|---|
| 1 | `RuntimePublicationOrchestrator.h:279`: `kMaxDeferredRetries = 10 → 2` + corrected comment. | Unchanged; safe (value-only). |
| 2 | `RuntimePublicationOrchestrator.h:32-37`: add `bool recoveryArmed{false}`. | Unchanged; structurally safe (whole-struct replace). |
| 3 | `RuntimePublicationOrchestrator.h:283`: delete `deferredRecoveryRearmed_` (dead code). | Unchanged; verified zero reads. |
| 4 | `AudioEngine.Timer.cpp:1724`: remove `resetDeferredRetryBudget()` call; keep atomic `setRecoveryPublishSeq` + mutex-guarded `publishRetryReady` + `notify`. | **BLOCKED** — must now *also* set a dedicated recovery-wake signal (Option D1/D3) **instead of** plain `publishRetryReady`, and the consumer at `RebuildDispatch.cpp:898-904` must forward it to `processDeferredAdmission`, which must branch on it (reset budget ↔ ordinary increment). |
| 5a | `processDeferredAdmission` (`RuntimePublicationOrchestrator.cpp:636-663`): accept wake-provenance arg; stamp `recoveryArmed` only on recovery wake; call `resetDeferredRetryBudget()` (now rebuild-thread) only on recovery wake. | **BLOCKED** until Option chosen. |
| 5b | Replace `resetDeferredRetryBudget()` body with a jassert-gated (rebuild-thread) variant. | New sub-step; `jassert(rebuildThreadId)` to harden the relocated reset. |

---

## 11. Lock File

- `publishRetryReady` producers: `AudioEngine.Threading.cpp:286` (ordinary, CoordinatorLoop thread), `AudioEngine.Timer.cpp:1744` (recovery, MessageThread). ✓
- `publishRetryReady` consumer: `AudioEngine.RebuildDispatch.cpp:879-880` → `doDeferredPublish` → `processDeferredAdmission()` `cpp:904` (no provenance arg). ✓
- `processDeferredAdmission` reads only `hasDeferred_` (`RuntimePublicationOrchestrator.cpp:639`); never reads `recoveryPublishSeq()`/`lastRecoveryPublishSeq_`/`recoveryPending`. ✓
- `recoveryPublishSeq()` (`RuntimePublicationOrchestrator.h:167-169`) has **zero** callers in `src/`. ✓
- `resetDeferredRetryBudget()` definition `h:171-174`; sole caller `AudioEngine.Timer.cpp:1724` (MessageThread, unguarded). ✓
- `clearDeferredForShutdown()` (`RuntimePublicationOrchestrator.cpp:534-560`) plain-writes retry members at `cpp:548-549`; MessageThread callers: `AudioEngine.Timer.cpp:1642, 1804, 1824`. ✓
- `recoveryPending` (`AudioEngine.h:2711`) is the ISR-level quarantined-DSP rebuild intent (consumed `RebuildDispatch.cpp:958`, pops `recoveryIntentQueue_`), **not** the D135 deferred-retry recovery — separate subsystem, not a provenance signal for `processDeferredAdmission`. ✓
- `DeferredPublishSlot` struct-replace-at-emplace (`cpp:505`), single aggregate-init site → defaulted `recoveryArmed{false}` cannot leak across overwrite (N3). ✓

---

## 12. D135-5 Gate

| Judgment item | Required | Evidence | Verdict |
|---|---|---|---|
| Recovery / ordinary wake distinction | Fully distinguishable | §2.1 (two producers, one blind flag), §4 (consumer reads no provenance) | **FAIL — NOT distinguishable** |
| Ordinary retry must not reset budget | Enforced | §5.4 (blind consume→reset collapses kMax=2) | **Cannot be enforced** without a signal |
| Recovery retry may reset budget on rebuild thread | Rebuild-thread only | §6 (budget reset relocatable; Timer site removed) | **OK once gated** |
| Timer thread touches no plain retry state | Zero plain access | §6 (Timer.cpp:1724 `resetDeferredRetryBudget` + three `clearDeferredForShutdown`) — must remove *all four* | **Must remove all four MessageThread writers** |
| `finishView()` | RebuildThread only | verified cpp:591 jassert (D135-4) | OK |
| `recoveryArmed` | slot lifecycle alignment | §7.1 (struct-replace ⇒ safe) | OK on placement; **gated on §7.2 stamp trigger** |
| Overwrite (N3) | no stale `recoveryArmed` leak | §7.1 | OK structurally |
| kMax=2 ordinary-retry upper bound | actually maintained | §5.3 (ordinary exhaustion trace) | **Violated by blind reset** |
| C4 (retry-identity conflation) | explicitly scoped out | D135-4 §9 | out of scope; note C4 is orthogonal (generation-key conflation), not the provenance defect |

**Overall Phase-1 verdict: NOT GO.** The D135-4 Phase-1 plan assumes a provenance that the consumer does not possess. Before Edit #4/#5a can be written, a wake-provenance signal (Option D1/D3/D4 preferred) must be selected, written at the producer side, and forwarded into `processDeferredAdmission`. This document is read-only; **no source was changed**.

# D135-6 — Recovery Wake Provenance Mechanism Design Audit

**Status:** READ-ONLY DESIGN AUDIT — zero production source changes.
**Scope:** Select one recovery-wake provenance mechanism so `processDeferredAdmission` can deterministically distinguish an ordinary Deferred retry wake from a Recovery crossfade-timeout wake.
**Gate:** Phase-1 source edits remain **prohibited** until D135-6 selects and proves the mechanism.
**Verified against:** live source tree + `ConvoPeq.md` snapshot. All cited lines confirmed consistent across both.

---

## 1. Context

### 1.1 Deferred-publish single-slot state machine

```
DeferredFadingActive → submitPublishRequest → enqueueDeferred (whole-struct slot replace)
→ rebuild-thread processDeferredAdmission → peekDeferred (non-flipping)
→ evaluateDeferred (4-level: Shutdown→TTL→Generation→Sequence; returns Ready/Discard)
→ consume()/discard() (both end in owner_->finishView())
→ finishView() (sole ownership-release mouth: deferredSlot_.reset() + hasDeferred_=false + telemetry, jassert rebuild-thread)
```

- **Single-slot** (`DeferredPublishSlot` in `std::optional<DeferredPublishSlot> deferredSlot_`).
- `enqueueDeferred` performs aggregate-struct replacement (INV-DEFERRED-2).
- `peekDeferred` does **not** flip `hasDeferred_`; only `finishView()` releases ownership.

### 1.2 Retry budget

- `deferredRetryGeneration_` (plain `int`) + `deferredRetryCount_` (plain `uint8_t`).
- Incremented at `enqueueDeferred` (`RuntimePublicationOrchestrator.cpp:472-478`).
- `RetryExhaustedDiscard` at `RuntimePublicationOrchestrator.cpp:489-502` returns **without** assigning the slot (`hasDeferred_` stays false → obligation lost).
- Live `kMaxDeferredRetries = 10` at `RuntimePublicationOrchestrator.h:280`, with a stale comment "2 回再駆動許容、3 回目で諦". Spec/binary value = 2.

> **Note on kMax staleness:** D135-5 established `kMaxDeferredRetries` is semantically 2 (spec) but live value is 10. The stale comment is a documentation defect, not a logic defect. D135-6 does not change the constant's value; that is a Phase-1 edit (P1, below). D135-6's mechanism choice is value-independent.

---

## 2. Wake-provenance gap (the problem)

### 2.1 The single shared flag

`publishRetryReady` (`AudioEngine.h:2718`) is a single `bool`, protected by `rebuildMutex`. It is written `true` by **two** producers:

| Producer | File | Line (live) | Provenance stamped? |
|---|---|---|---|
| Ordinary retry (CoordinatorLoop) | `AudioEngine.Threading.cpp` | 286 | **No** |
| Recovery crossfade-timeout (MessageThread) | `AudioEngine.Timer.cpp` | 1744 | **No** |

Both writers acquire `rebuildMutex`, set `publishRetryReady = true`, then call `rebuildCV.notify_one()`.

### 2.2 The consumption merge (provenance lost)

At the rebuild-thread wake site (`AudioEngine.RebuildDispatch.cpp:879-880`):

```cpp
doDeferredPublish = publishRetryReady;   // [879] provenance-lost merge
publishRetryReady = false;               // [880]
```

Then at `AudioEngine.RebuildDispatch.cpp:898-904`:

```cpp
if (doDeferredPublish && runtimeOrchestrator_ != nullptr)
{
    runtimeOrchestrator_->processDeferredAdmission();   // [904] no provenance arg
}
```

`processDeferredAdmission()` at `RuntimePublicationOrchestrator.cpp:636` takes **zero arguments** and reads only `hasDeferred_` (line 639). It has no way to know whether the wake came from an ordinary retry or a recovery crossfade-timeout.

### 2.3 The write-only seq latch

`recoveryPublishSeq()` (`RuntimePublicationOrchestrator.h:167-169`) is called **zero times** on the rebuild side. `lastRecoveryPublishSeq_` (`RuntimePublicationOrchestrator.h:281`) is:
- **Set** at `AudioEngine.Timer.cpp:1723` (recovery handler, MessageThread).
- **Cleared** at `RuntimePublicationOrchestrator.cpp:551` (`clearDeferredForShutdown`).
- **Never consumed** on the rebuild thread.

This is a dead/lost-provenance signal — written at production but never read where it would matter.

### 2.4 Snapshot consistency check

`ConvoPeq.md` snapshot lines for all audited sites match live source exactly:
- `ConvoPeq.md:39260-39322` ↔ `AudioEngine.RebuildDispatch.cpp:840-910` ✓
- `ConvoPeq.md:42555-42668` ↔ `AudioEngine.Timer.cpp:1669-1750` ✓
- `ConvoPeq.md:66138-66312` ↔ `RuntimePublicationOrchestrator.h/.cpp` ✓

D135-5's baseline findings are **not contradicted** by the current code.

---

## 3. Option evaluation

### D1: Dedicated atomic flag `recoveryRetryReady`

**Mechanism:** Add `std::atomic<bool> recoveryRetryReady{false}` to `AudioEngine` (same `rebuildMutex`-guarded region as `publishRetryReady`). Recovery handler becomes the **sole writer** (sets `recoveryRetryReady = true` alongside `publishRetryReady = true`). At the consumer merge site (`RebuildDispatch.cpp:879-880`), the rebuild thread reads and clears `recoveryRetryReady` to derive a local `wasRecoveryWake` boolean, then forwards it to `processDeferredAdmission(wasRecoveryWake)`.

**Evaluation:**
- ✅ Both wake producers and the consumer merge are co-located: writers at `Threading.cpp:286` and `Timer.cpp:1744`; consumer merge at `RebuildDispatch.cpp:879-880`; immediate call at line 904. A local `wasRecoveryWake` variable can be forwarded as the arg.
- ✅ Recovery is the sole writer of `recoveryRetryReady` — ordinary retry does **not** touch it. This is a strict ownership invariant (like the existing `recoveryPending` at `AudioEngine.h:2711`, which is ISR-level and separate).
- ✅ `recoveryRetryReady` is cleared at consumption under `rebuildMutex` (same locked section as `publishRetryReady` at lines 879-880), eliminating stale-signal races.
- ✅ No CV predicate rewrite required (D3 not needed).
- ✅ Preserves the ordinary-exhaustion trace (count 0→1→2→starved) — budget reset fires only on recovery wakes.
- ✅ Recovery precedence is deterministic: if `recoveryRetryReady` is true, budget **must** reset regardless of `publishRetryReady`'s source.

**Rejected variant:** Combining D1's flag with the existing seq latch (`lastRecoveryPublishSeq_`). The seq is set under the MessageThread's RCU reader path (Timer.cpp:1722-1723) and read by the rebuild thread — a seq value alone cannot distinguish writer provenance under CV-notify coalescing. D1's boolean flag is sufficient and minimal.

### D2: Consume-on-read seq latch (`lastRecoveryPublishSeq_`)

**Mechanism:** Make `recoveryPublishSeq()` consume (clear) `lastRecoveryPublishSeq_` on read. The rebuild thread calls it after wake; if non-zero, the wake is classified as recovery.

**Evaluation:**
- ❌ **Unresolvable coalescing misclassification.** The CV predicate at `AudioEngine.RebuildDispatch.cpp:849-854` waits on `hasPendingTask || publishRetryReady || recoveryPending || rebuildThreadShouldExit`. Under CV re-check semantics, a single wakeup can serve **two batched flag sets** (e.g., ordinary retry and recovery fire in the same notification window). If the first `processDeferredAdmission` call consumes the seq (clears it to 0), a second batched call in the same wake will read 0 and misclassify a recovery wake as ordinary — a lost-provenance defect.
- ❌ `lastRecoveryPublishSeq_` is an `atomic<PublicationSequenceId>` (sequence number), not a per-wake latch. Clearing it on consume conflates "which seq was this" with "was this a recovery wake" — two orthogonal signals jammed into one field.
- ❌ The seq is set at Timer.cpp:1723 **before** `publishRetryReady = true` at line 1744, but both are read at the **same** merge site (879-880). A seq-based read-after-consume at a downstream point cannot retroactively fix the co-located merge.

**Decision: REJECTED.**

### D3: Enum-based wake reason in the CV predicate

**Mechanism:** Replace the `bool publishRetryReady` with an enum `{Idle, OrdinaryRetry, RecoveryWake}` and rewrite the CV predicate to match on the enum.

**Evaluation:**
- ⚠️ Larger footprint than needed. Requires CV predicate rewrite (`RebuildDispatch.cpp:849-854`), all producer writers (Threading.cpp:286, Timer.cpp:1744), and the consumer merge (879-880) to change from bool to enum.
- ⚠️ The predicate currently uses `publishRetryReady` as a simple boolean guard to prevent busy-looping during `DeferredFadingActive`. Switching to an enum adds complexity (Idle state must be distinguishable from "no pending work").
- Per D135-5's preference order, D3 was the preferred approach for a *full* wake-provenance redesign. For the **scoped** D135-6 objective (deterministically distinguish ordinary vs. recovery in `processDeferredAdmission`), D1 achieves the same correctness with strictly fewer edits and no predicate change.

**Decision: REJECTED for this scope.** D3 is reserved for Phase 2 (if a broader wake-reason taxonomy becomes necessary).

---

## 4. Selected mechanism: D1

### 4.1 Rationale (summary)

D1 is the **minimal correct** mechanism. It adds exactly one new `std::atomic<bool>` field with a single recovery-only writer, clears it at the co-located consumer merge, and forwards a local boolean to `processDeferredAdmission`. This satisfies the D135-2 re-run gate (recovery-redrive > 0 must be observable) and the D135-1 ordinary-exhaustion invariant simultaneously.

### 4.2 Concurrent wake coalescing precedence

**Definition:** `wasRecoveryWake = recoveryRetryReady` (read-and-clear at the merge site).

**Precedence rule:** Recovery takes precedence over ordinary:
- `recoveryRetryReady == true` ⇒ budget **must** reset (regardless of `publishRetryReady`'s provenance).
- `recoveryRetryReady == false` AND `publishRetryReady == true` ⇒ ordinary retry, increment budget.
- `recoveryRetryReady == true` AND `publishRetryReady == true` ⇒ recovery wins (the `publishRetryReady=true` was redundant; recovery is the dominant intent).

This is the minimal deterministic definition derived from existing semantics. The recovery handler resets the budget **before** setting `publishRetryReady = true` (Timer.cpp:1724 before 1744), so if the rebuild thread consumes in between, the budget is already reset — and under D1, the rebuild thread reads `recoveryRetryReady` (set at the same scope as `publishRetryReady` at 1744) and applies its own reset, making the Timer.cpp:1724 `resetDeferredRetryBudget()` call **redundant** under D1 (P5 removes it to eliminate the cross-thread plain-write race).

### 4.3 Lost/stale wake verification

**CV predicate** (`AudioEngine.RebuildDispatch.cpp:849-854`):

```cpp
rebuildCV.wait(lock, [this] {
    return hasPendingTask
        || publishRetryReady
        || recoveryPending
        || convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire);
});
```

- The predicate already includes `publishRetryReady`. Under D1, `recoveryRetryReady` is set **under the same `rebuildMutex` lock** as `publishRetryReady` (both writers at Threading.cpp:286 and Timer.cpp:1744 acquire `rebuildMutex` before setting). Therefore, if `publishRetryReady` wakes the CV, `recoveryRetryReady` is already visible in the same locked scope.
- **Case A (ordinary retry only):** `publishRetryReady=true`, `recoveryRetryReady=false`. Consumer derives `wasRecoveryWake=false`. Budget increments. ✓
- **Case B (recovery only):** `publishRetryReady=true`, `recoveryRetryReady=true`. Consumer derives `wasRecoveryWake=true`. Budget resets. ✓
- **Case C (both batched):** CV fires once for both. Consumer reads `wasRecoveryWake=true`. Budget resets. Second batched iteration (if any) reads `wasRecoveryWake=false` (already cleared). This is correct: the recovery intent is satisfied by the first consumption. ✓
- **Case D (spurious wake):** Predicate re-checks under lock. Both flags false → `wasRecoveryWake=false`. No budget mutation. ✓
- **Slot-gone case (clearDeferredForShutdown interleaved):** If `clearDeferredForShutdown` runs between the wake and `processDeferredAdmission`, `hasDeferred_` is false and `processDeferredAdmission` returns early at line 639-640. `wasRecoveryWake` is irrelevant — no budget mutation occurs. ✓

**Stale-signal elimination for D1:** `recoveryRetryReady` is cleared at the same merge site as `publishRetryReady` (`RebuildDispatch.cpp:879-880`), under the same `rebuildMutex` lock. No stale signal can persist past consumption.

### 4.4 `lastRecoveryPublishSeq_` handling

Under D1, `lastRecoveryPublishSeq_` is **retained** for diagnostic/telemetry purposes (D135-1 §5 uses it for `[D135] recovery-redrive` logging at Timer.cpp:1733-1739). It is **not** used for wake-provenance discrimination (that role is now `recoveryRetryReady`). The sequence remains useful as a correlation ID for log/trace analysis.

The audit confirms: `recoveryPublishSeq()` has zero rebuild-thread callers (only the diagLog at Timer.cpp:1733-1739 reads the seq value directly, not via `recoveryPublishSeq()`). Under D1, no change is needed to the seq field — D1's boolean flag handles provenance; the seq handles correlation.

---

## 5. Ownership audit: `clearDeferredForShutdown()` and `resetDeferredRetryBudget()`

### 5.1 `clearDeferredForShutdown()`

**Definition:** `RuntimePublicationOrchestrator.cpp:534-560`

**Members touched (all plain writes):**
- `deferredSlot_.reset()` (line 543)
- `hasDeferred_ = false` via `publishAtomic` (line 544)
- `deferredRetryGeneration_ = 0` (line 548)
- `deferredRetryCount_ = 0` (line 549)
- `deferredRecoveryRearmed_ = false` (line 550)
- `lastRecoveryPublishSeq_ = 0` via `publishAtomic` (line 551)

**Callers (all confirmed live + snapshot):**

| Caller | File | Line (live) | Thread | jassert? |
|---|---|---|---|---|
| ReleaseResources (shutdown) | `AudioEngine.Processing.ReleaseResources.cpp` | 359 | MessageThread (shutdown path, safe) | No |
| Timer stall path | `AudioEngine.Timer.cpp` | 1642 | MessageThread | **No** |
| Timer Recover | `AudioEngine.Timer.cpp` | 1804 | MessageThread | **No** |
| Timer Restore | `AudioEngine.Timer.cpp` | 1824 | MessageThread | **No** |

**Contract violation:** `clearDeferredForShutdown()` performs plain writes to `deferredRetryGeneration_` and `deferredRetryCount_` (lines 548-549) — members documented as rebuild-thread-only (single-owner). Three of four callers are on the **MessageThread**, which violates the single-owner contract. The function has **no rebuild-thread jassert** (unlike `peekDeferred` at line 566, `finishView` at line 591, and `processDeferredAdmission` at line 638, all of which jassert `std::this_thread::get_id() == engine_.rebuildThreadId()`).

**D135-5 finding confirmed:** This is a required Phase-1 follow-up edit (P8), not a D135-6 blocker on the provenance mechanism itself. The provenance flag (`recoveryRetryReady`) lives on `AudioEngine`, not the orchestrator, and is written under `rebuildMutex` — it does not change the `clearDeferredForShutdown()` ownership analysis.

### 5.2 `resetDeferredRetryBudget()`

**Definition:** `RuntimePublicationOrchestrator.h:171-174` (inline):

```cpp
void resetDeferredRetryBudget() noexcept {
    deferredRetryGeneration_ = 0;
    deferredRetryCount_ = 0;
}
```

**Callers:**

| Caller | File | Line (live) | Thread |
|---|---|---|---|
| Timer recovery handler | `AudioEngine.Timer.cpp` | 1724 | MessageThread |

**Contract violation:** Writes rebuild-thread-only plain members (`deferredRetryGeneration_`, `deferredRetryCount_`) from the MessageThread. Same ownership violation as `clearDeferredForShutdown()`.

**Under D1:** This call becomes **redundant** — the rebuild thread will reset the budget itself when it consumes a recovery wake (P7). P5 removes the `resetDeferredRetryBudget()` call from Timer.cpp:1724, eliminating the cross-thread plain-write race entirely.

**P9 (sub-P8):** A `jassert(std::this_thread::get_id() == engine_.rebuildThreadId())` is added inside `resetDeferredRetryBudget()` itself, making any future cross-thread call a fail-fast assertion rather than a silent data race.

---

## 6. Phase-1 edit redefinition (by necessity, not "fit into 5")

D135-4's 5-edit plan is insufficient for D1's requirements. The following edit list is **required** under D1. Each edit is justified by a specific defect, not by an arbitrary budget.

| Phase | Edit | File | Defect addressed |
|---|---|---|---|
| **P1** | `kMaxDeferredRetries`: `10 → 2` | `RuntimePublicationOrchestrator.h:280` | Stale value/spec mismatch (live=10, spec=2) |
| **P2** | Add `bool recoveryArmed{false}` to `DeferredPublishSlot` | `RuntimePublicationOrchestrator.h:32-38` | Slot-level recovery provenance for per-obligation tracking |
| **P3** | Delete `deferredRecoveryRearmed_` dead member | `RuntimePublicationOrchestrator.h:283` | Dead code (zero readers) |
| **P4** | Add `std::atomic<bool> recoveryRetryReady{false}` to `AudioEngine` | `AudioEngine.h` (near `publishRetryReady`) | **Core D1: provenance flag, recovery-only writer** |
| **P5** | Recovery handler: remove `resetDeferredRetryBudget()` call | `AudioEngine.Timer.cpp:1724` | Eliminate cross-thread plain-write race (budget reset moves to rebuild thread) |
| **P6** | Consumer merge: read+clear `recoveryRetryReady` → local `wasRecoveryWake`; forward to `processDeferredAdmission` | `AudioEngine.RebuildDispatch.cpp:879-904` | **Wire D1: forward provenance through consumer** |
| **P7** | `processDeferredAdmission(bool wasRecoveryWake)`: stamp `recoveryArmed` on slot + reset budget only when `wasRecoveryWake==true` | `RuntimePublicationOrchestrator.cpp:636-663` + `.h` declaration | **Wire D1: conditional budget reset on rebuild thread** |
| **P8** | `clearDeferredForShutdown()`: add rebuild-thread jassert; route MessageThread callers (Timer.cpp:1642/1804/1824) to rebuild-thread drain | `RuntimePublicationOrchestrator.cpp:534-560` + Timer call sites | Single-owner contract repair |
| **P9** | `resetDeferredRetryBudget()`: add rebuild-thread jassert | `RuntimePublicationOrchestrator.h:171-174` | Defensive fail-fast for any future cross-thread call |

**Edit count: 9 (P1–P9), not 5.** This is an increase from D135-4's plan because D1's correctness requires closing the `clearDeferredForShutdown()` ownership gap (P8–P9), which D135-5's observation was elevated to a Phase-1 requirement during this audit.

### 6.1 Non-goals for Phase 1 (scoping)

- The `recoveryPublishSeq()` write-only latch (`RuntimePublicationOrchestrator.h:167-169`) is **retained** as a correlation/log ID. Its zero-caller status is documented but not deleted in Phase 1 — it is read by the diagLog path at Timer.cpp:1733-1739 via `recoverySeq` (a local), not via `recoveryPublishSeq()`. Removing it is a Phase-2 cleanup.
- `deferredRecoveryReararmed_` (P3 deletion) is dead code with zero readers across the entire codebase (verified via grep). Its removal is safe and required for slot-struct cleanliness.

---

## 7. D135-2 re-run compatibility

D135-2's re-run gate at `--cli-intent-burst-interval-ms 4000` expects `recovery-redrive > 0`. Under D1:

- The ordinary-exhaustion trace (count 0→1→2→starved, recovery-redrive=0) is **preserved** — budget reset fires only on recovery wakes (`wasRecoveryWake==true`), not ordinary retries.
- The recovery-redrive trace (recovery wake → `wasRecoveryWake==true` → budget reset → fresh retries → `recovery-redrive > 0`) becomes **deterministically observable** once the provenance signal exists.
- No conflict with D135-2's existing test expectations. The D135-2 harness reads `recovery-redrive` from the `[D135] recovery-redrive` diagLog at Timer.cpp:1733-1739, which reads `recoverySeq` (a local) — unaffected by D1's mechanism change.

---

## 8. kMaxDeferredRetries = 2 reachability proof (under D1)

- **Ordinary retry path:** `enqueueDeferred` (cpp:472-478) increments `deferredRetryCount_` with **no budget reset** (P5 removed the Timer.cpp:1724 reset; P7 only resets on `wasRecoveryWake==true`). `RetryExhaustedDiscard` (cpp:489-502) is reachable after 3 attempts (count 0→1→2→starved at kMax=2). ✓
- **Recovery path:** `wasRecoveryWake==true` at P7 resets `deferredRetryCount_ = 0` and `deferredRetryGeneration_` advances, granting 2 fresh retries. ✓

Both invariants hold under D1.

---

## 9. GO conditions checklist

| # | Condition | Satisfied under D1? | Evidence |
|---|---|---|---|
| GO-1 | Consumer can deterministically distinguish ordinary retry wake from recovery wake | ✅ YES | P4 `recoveryRetryReady` (recovery-only writer) + P6 read-and-clear at co-located merge (RebuildDispatch.cpp:879-880) |
| GO-2 | `wasRecoveryWake` definition is deterministic under CV-notify coalescing | ✅ YES | §4.2: recovery precedence; §4.3 Case C (both batched → recovery wins on first consumption) |
| GO-3 | No stale-signal race on `recoveryRetryReady` | ✅ YES | §4.3: cleared at same locked scope as `publishRetryReady` (879-880) |
| GO-4 | D2 (`lastRecoveryPublishSeq_` consume-on-read) rejected for correct reason | ✅ YES | §3: coalescing misclassification proof |
| GO-5 | D3 enum predicate rewrite not required | ✅ YES | §3: D1 achieves same correctness with fewer edits |
| GO-6 | Every `clearDeferredForShutdown()` caller audited (thread, contract, effects) | ✅ YES | §5.1: 4 callers, 3 MessageThread violations (P8 required) |
| GO-7 | `resetDeferredRetryBudget()` ownership audited | ✅ YES | §5.2: 1 MessageThread caller (Timer.cpp:1724) → P5 removes, P9 adds jassert |
| GO-8 | Phase-1 edit list redefined by necessity (P1–P9), not arbitrary | ✅ YES | §6: 9 edits justified by specific defects |
| GO-9 | D135-2 re-run compatibility verified | ✅ YES | §7: ordinary-exhaustion trace preserved; recovery-redrive gate valid |
| GO-10 | kMax=2 reachability proof holds under D1 | ✅ YES | §8: ordinary path no-reset; recovery path reset-on-wake |
| GO-11 | Live source + ConvoPeq.md snapshot consistent | ✅ YES | §2.4: all cited lines verified across both |
| GO-12 | Zero production source changes in this audit | ✅ YES | This document is read-only design. Edits begin only in Phase 1 (post-gate). |

---

## 10. Conclusion

**D1 is selected.** It is the minimal correct mechanism: one new `std::atomic<bool>` (`recoveryRetryReady`), recovery-only writer, rebuild-thread consumer clearing at the co-located merge site (RebuildDispatch.cpp:879-880), forwarded as `wasRecoveryWake` to `processDeferredAdmission(bool wasRecoveryWake)`.

The wake-decision site and the `processDeferredAdmission` call are co-located in the same locked section + immediate unlocked scope (RebuildDispatch.cpp:879-904), enabling both the arg-forward pattern (P6) and the flag-read pattern (P4) within D1.

`clearDeferredForShutdown()` ownership repair (P8–P9) is a required Phase-1 follow-up, elevated from D135-5's "race class" observation to a must-fix for single-owner compliance. It does not alter D1's mechanism selection.

Phase-1 source edits remain **prohibited** until this design is accepted. The 9-edit plan (P1–P9) is the complete, necessity-driven implementation sequence.

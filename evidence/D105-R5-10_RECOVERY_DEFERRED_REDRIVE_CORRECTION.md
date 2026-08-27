# D105-R5-10 — Recovery Logical Obligation: Deferred Re-drive Correction

**Status:** PASSED ✅ (D105-BUILD-1 resolved → Debug+Release build green; C11–C16 PASS; 256-stress `ISRSoakTests` PASS; D105-R10 read-only audit PASS)
**Closes:** D105-R9 R8-#4 (FAIL: "延期された recovery obligation の再配送 (re-drive) が未実装")
**Closes:** D105-R9 R8-#4 (FAIL: "延期された recovery obligation の再配送 (re-drive) が未実装")
**Prerequisite:** D105-R5-9 (MUST-1..4) already applied; single `+1`/`−1` authority intact (`tryInsert` h:339-only `+1`, `resolve` h:357-only `−1`).

---

## 1. Problem (R8-#4)

A recovery obligation that is admitted (L incremented) but then **deferred** — because BOTH delivery
resources are unavailable (transport `recoveryIntentQueue_` full at 256 AND the single durable slot
`pendingRecoveryAdmission_` occupied by a *different* live obligation) — was left with **no delivery
representation** and was **never rediscovered**. This is a silent recovery loss (violates INV-X1-2:
"queue full ≠ Recovery lost").

Deferred condition (reached only when queue capacity 256 is filled, which requires 256 coalesced
re-submissions of one target — see §3):

```
state == Live  &&  delivery == None   (and recoveryRetryDeferredCount_ incremented)
```

## 2. Design (ΔL = 0 — hard constraint)

Added a per-obligation **delivery residency** field, independent of `ObligationState`:

```cpp
enum class ObligationDeliveryState : uint8_t { None, Transport, Durable };
// LogicalRecoveryObligation.delivery  { None=deferred, Transport=in queue, Durable=in durable slot }
```

- The re-drive **never** inserts a new obligation (`tryInsert` untouched) and **never** terminates one
  (`resolve` untouched). It only re-attaches a delivery representation to an **existing Live** obligation,
  preserving its `id` + `buildSource` (no new `LogicalRecoveryObligationId` issued).
- `redriveDeferredRecovery(id)` rebuilds a `RecoveryIntent` from the table slot and:
  - durable slot free → admit to `pendingRecoveryAdmission_` (`delivery = Durable`);
  - else transport queue has space → `recoveryIntentQueue_.push` (`delivery = Transport`);
  - else both busy → leave `delivery = None`, `recoveryRetryRedriveFailureCount_++` (still Live, ΔL=0, retried later).
- `redriveDeferredRecoveryObligations()` scans the 32-slot table; for each `Live && delivery==None` it calls
  `redriveDeferredRecovery(id)`. **§5 guarantee:** it skips any obligation whose `delivery != None`, so a
  Transport/Durable obligation is **never re-sent** (no duplicate delivery representation).

### Trigger points (SPSC-safe — producer/CoordinatorLoop thread only)
1. `submitRecoveryRequest()` start (after the shutdown gate) — opportunistic, prompt on activity.
2. `AudioEngine::runCoordinatorPhase()` (the existing 1 ms `CoordinatorLoop` tick, option C in the spec) —
   always-on rediscovery even when no new submissions occur.

Both run on the **CoordinatorLoop (producer)** thread, the sole writer of `recoveryIntentQueue_` /
`pendingRecoveryAdmission_` ⇒ no SPSC violation.

### Key design correction during implementation
The first draft added an early-return in `submitRecoveryRequest` that skipped re-push when an obligation
already had a delivery representation. This **broke the queue-fill mechanism** that the existing
`testRecoveryDurableAdmission` relies on (it fills the 256 queue by coalescing 256 re-submissions of one
target onto a single obligation). It also made the deferred state **unreachable**. The early-return was
**reverted**: the §5 "no double delivery" rule is enforced by the **re-drive's own `delivery==None` guard**,
not by suppressing the normal coalesce re-push. The deferred path is now genuinely reachable
(256 coalesced intents → 257th distinct target → durable → 258th distinct target → deferred).

## 3. Files changed

- `src/audioengine/ISRRuntimePublicationCoordinator.h`
  - `enum class ObligationDeliveryState` (after `ObligationState`).
  - `LogicalRecoveryObligation::delivery` field (init `None`).
  - `tryInsert`: reset `delivery = None` on slot reuse.
  - Public decls `redriveDeferredRecoveryObligations()` / `redriveDeferredRecovery(id)`.
  - Telemetry accessors `recoveryRetryRedriveCount()` / `recoveryRetryRedriveFailureCount()`.
  - Members `recoveryRetryRedriveCount_` / `recoveryRetryRedriveFailureCount_`.
- `src/audioengine/ISRRuntimePublicationCoordinator.cpp`
  - `submitRecoveryRequest`: set `delivery = Transport` on queue push success, `= Durable` on durable
    admission, `= None` in the defer branch (MUST-2 / D105-R5-9 defer path).
  - `redriveDeferredRecoveryObligations()` + `redriveDeferredRecovery(id)` implementations.
  - Trigger call at `submitRecoveryRequest` start.
- `src/audioengine/AudioEngine.Threading.cpp`
  - `runCoordinatorPhase()`: `runtimePublicationBridge_.redriveDeferredRecoveryObligations();` each tick.
- `src/tests/ISRSemanticValidationTests.cpp`
  - `fillRecoveryQueue()` helper + tests **C11–C16** (see §4), registered in `main()`.

## 4. Tests C11–C16 (added, registered in `main()`)

| Test | Verifies |
|------|----------|
| C11 | Deferred obligation redriven onto durable slot when it frees (ΔL=0). |
| C12 | Redrive idempotent — second redrive on delivered obligation creates no duplicate. |
| C13 | Redrive prefers durable, falls back to transport when durable busy. |
| C14 | Redrive when both resources busy stays deferred (ΔL=0, failure counted), recovers later. |
| C15 | Redrive never changes live obligation count (no +1 / no −1). |
| C16 | Deferred + same-key re-submission coalesces (no new id, L unchanged, single representation). |

All six use the reachable scenario: `fillRecoveryQueue` (256 coalesced intents of target-1 → queue full,
O1 transport) → target-2 submit → durable (free slot) → target-3 submit → **deferred**
(`recoveryRetryDeferredCount_>=1`).

## 5. Invariant preservation (static verification)

- `kMaxLogicalRecoveryObligations = 32` unchanged; `RecoveryAdmissionTable` `tryInsert`/`resolve` untouched
  ⇒ exactly one `+1` (h:339) / one `−1` (h:357). No `liveCount_++/--` added.
- No `resolve()` call added; no `reclaimSlot()`/`findById` introduced.
- `pendingIntentCount_` semantics, queue capacity 256, and the single-slot `PendingRecoveryAdmission`
  unchanged.
- Re-drive does not issue a new `LogicalRecoveryObligationId`; obligation id + `buildSource` restored from slot.
- `kMaxSlots=256` in `ISRDSPQuarantine.h` / `ISRRetireRuntimeEx.h` are **DSP slots**, unrelated to L.

## 6. Verification status — PASSED

- **D105-BUILD-1 (resolved):** `AudioEngine.Commit.cpp:812` (`recoveryObligationId` undeclared) fixed
  by adding `std::uint64_t recoveryObligationId` param to `enqueuePublicationIntentForRuntimeCommit`
  (carried on `PublishRequest.recoveryObligationId` → identity chain preserved, no deletion). Additionally
  surfaced a **pre-existing R5-8/R5-9 desync**: `RecoveryIntent` aggregate (no ctor) has 5 fields
  `(handle, epoch, intentId, obligationId, buildSource)`, but the 3 construction sites positional-init'd
  4 fields — landing `buildSource` onto the `uint64_t obligationId` field (`C2440` at cpp:833/998/1047).
  Fixed by aligning the 3 aggregate-init lists to the 5-field layout (obligationId explicit in each);
  `takePendingRecoveryAdmission` now init's all 5 in correct order. **No R5-10 re-drive design change.**
  Debug + Release `ConvoPeq` builds: both EXIT 0.
- **D105-R10 (read-only audit PASS — see §8):**
  - R10-1 `ObligationDeliveryState`: None=deferred/Live, Transport=in-queue pending, Durable=in single
    durable slot. ✓
  - R10-2 ΔL=0: `redriveDeferredRecovery` issues **no** `tryInsert`/`resolve`/`liveCount_` (only re-attaches
    the existing obligation id at cpp:1019 — `intent.obligationId = obligationId`, no new id/obligation).
    Single +1/−1 authority (`tryInsert` h:339-only +1, `resolve` h:357-only −1) untouched. ✓
  - R10-3 no-double-delivery: `delivery != None → continue` scan guard (cpp:981) + idempotent `delivery != None
    → return` per-call (cpp:1007); coalesce branch early-returns when *this submission's* redrive already
    attached the representation (cpp:866). ✓
  - R10-4 queue-full cascade (target-1 ×256 → transport; target-2 → durable; target-3 → deferred) — verified by C11/C14. ✓
- **CTest `ISRSemanticValidationRejects` (C1–C16): PASS.**
- **`ISRSoakTests` (D105-R5-10 256-stress): PASS.**

**C16 correction note:** a first runtime pass surfaced C16 failing — `redrive` (invoked at `submitRecoveryRequest` start)
already enqueued the single transport representation for a deferred obligation, then the coalesced resubmit
re-pushed a *second* intent → 2 representations (violated single-representation). Fixed minimally by capturing
`wasDeferredBefore` (the target's `delivery==None` snapshot pre-redrive) and early-returning in the coalesce
branch when redrive already delivered in this call — preserving the `fillRecoveryQueue` re-push behavior
(O1 was already Transport pre-redrive ⇒ `wasDeferredBefore` false ⇒ still re-pushes to fill 256). **No R5-10
mechanism redesign**; the deferred re-drive trigger (submit-start + `redriveDeferredRecoveryObligations`),
durable slot, and id-passthrough are unchanged.

## 7. Next (done)

1. ✅ **D105-BUILD-1**: `AudioEngine.Commit.cpp:812` + `RecoveryIntent` 5-field aggregate desync resolved
   (identity chain preserved; no deletion; R5-10 untouched). Debug + Release `ConvoPeq` green.
2. ✅ **D105-R10**: read-only re-verification of R8-#4 — PASS (re-drive present at cpp:826 submit-start;
   triggers correct; ΔL=0).
3. ✅ CTest `ISRSemanticValidationRejects` (C1–C16) — PASS; `ISRSoakTests` (256-stress) — PASS.
4. ✅ R8-#4 flipped from FAIL → CLOSED.

## 8. R10 read-only audit checklist

- R10-1 `ObligationDeliveryState` semantics (None/Transport/Durable) — matches code (§2). ✅
- R10-2 `redriveDeferredRecovery` contains no `tryInsert`/`resolve`/`liveCount_`++/−; re-uses existing id. ✅
- R10-3 no-double-delivery: scan guard `delivery != None → continue` + per-call idempotent return + C16
  coalesce early-return. ✅
- R10-4 queue-full path reachable (target-1 ×256 → Durable → Deferred) — C11/C14 verify. ✅

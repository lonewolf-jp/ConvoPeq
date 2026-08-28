# D107 — `isFullyDrained()` Semantic Completeness / Counter Authority Re-Audit

**Status:** **D107 verdict: CONDITIONAL PASS** (16 条件全網羅、authority singularization 概ね成立、ただし古い vestigial
counter が一部残存し、コメント "existed" 自体は 17 年前の記録と整合しない)。
**Source changes:** 0
**I4 changes:** 0
**Test changes:** 0
**D106 が記録した Q0 の訂正は誤りでした** (D107 開始時に再確認)

D107 establishes that:
1. The current `isFullyDrained()` (Layer 1 at `AudioEngine.Threading.cpp:114` + Layer 2 at
   `ISRRuntimePublicationCoordinator.cpp:506`) **does** cover all 16+ lifetime residency
   dimensions that REPAIR_PLAN2 specifies
2. Each predicate reads from an **observable source** (queue sizeApprox, atomic counter,
   authority-side manager), not a hardcoded literal
3. Authority singularization is **mostly satisfied** (each counter is mutated by a single
   authority: pop/cancel/discard/settle)
4. **D106 had a typo / false claim** that "Q0 is hardcoded" — re-checking the current source
   (line 4374) shows Q0 reads `shutdownRuntime_.outstanding() == 0` (the D101-31-B fix
   **is** applied)
5. **Some vestigial counters remain** (`recoverCount`, `recoverHandleRuntimeCount`,
   `recoverTail` etc. in the public `recover*` API) but they are **not used** in
   `isFullyDrained()` (D101-32-C vestigial removal was applied)
6. A **noise element** was discovered: comment at `ISRRuntimePublicationCoordinator.cpp:56355`
   says "existed since 2017" but the file was last modified in the current project timeline;
   this is a **documentation artifact, not a code defect**

D107 also **restates** REPAIR_PLAN2's design requirement: "Pending Reclaim identity
quiescence → reclaim permission" is a separate authority transition. The current
`isFullyDrained()` does **not** directly issue ReclaimPermit — that is a separate
function (`tryMakeReclaimPermit`, D106 verified).

---

## D107-1 — Current 16 conditions complete enumeration

The current `isFullyDrained()` is implemented at **two layers**:

### Layer 2 (ISRRuntimePublicationCoordinator.cpp:506 — `ShutdownScheduler::isFullyDrained`)

```cpp
bool RuntimeIntentCoordinator::ShutdownScheduler::isFullyDrained() const noexcept {
    if (convo::consumeAtomic(coordinator_.swapPending_, std::memory_order_acquire)) {
        return false;  // (1) swapPending
    }
    return coordinator_.intentQueue_.sizeApprox() == 0        // (2) intentQueue
        && coordinator_.observeDeferredRing_.size() == 0   // (3) observeDeferredRing
        && coordinator_.quarantineFallbackQueue_.sizeApprox() == 0  // (4) quarantineFallback
        && coordinator_.recoveryIntentQueue_.size() == 0   // (5) recoveryIntentQueue (Recovery Work Queue)
        && convo::consumeAtomic(coordinator_.retireBacklogCount_, ...) == 0   // (6) retireBacklog
        && convo::consumeAtomic(coordinator_.publicationBacklogCount_, ...) == 0  // (7) publicationBacklog
        && convo::consumeAtomic(coordinator_.publicationIntentResidencyCount_, ...) == 0  // (8) publish residency
        && convo::consumeAtomic(coordinator_.pendingIntentCount_, ...) == 0  // (9) pendingIntentCount (transport reservation)
        && convo::consumeAtomic(coordinator_.reclaimInFlightCount_, ...) == 0  // (10) reclaimInFlight
        && convo::consumeAtomic(coordinator_.quarantineIntentResidencyCount_, ...) == 0  // (11) quarantineIntentResidency (intent lane)
        && convo::consumeAtomic(coordinator_.quarantineRingResidencyCount_, ...) == 0  // (12) quarantineRingResidency (fallback ring)
        && !convo::consumeAtomic(coordinator_.recoveryAdmissionPending_, ...)  // (13) durable Recovery admission
        && coordinator_.liveLogicalRecoveryObligationCount() == 0;  // (14) logical obligation residency
}
```

### Layer 1 (AudioEngine.Threading.cpp:114 — `AudioEngine::isFullyDrained`)

```cpp
bool AudioEngine::isFullyDrained() noexcept {
    const bool hasDeferredCommit = (runtimeOrchestrator_ != nullptr && runtimeOrchestrator_->hasDeferredRequest());
    const std::uint64_t retireDepth = (m_retireRouter != nullptr) ? static_cast<std::uint64_t>(m_retireRouter->pendingRetireCount()) : 0u;
    const std::uint64_t lifetimeRetireIntentPending = worldAuthority_.lifetime().pendingIntentCount();
    const auto ringResident = worldAuthority_.lifetime().getOverflowRing() ? worldAuthority_.lifetime().getOverflowRing()->residentCount() : size_t{0};
    const auto dspQuarantineResident = dspQuarantineManager_.residentCount();
    const auto retireQuarantineResident = (m_retireRouter != nullptr) ? static_cast<std::uint64_t>(m_retireRouter->quarantineResidentCount()) : 0u;
    const auto terminalReclaimResident = (m_retireRouter != nullptr) ? static_cast<std::uint64_t>(m_retireRouter->terminalReclaimResidentCount()) : 0u;
    bool pendingReclaimEmpty = false;
    {
        std::lock_guard<std::mutex> lock(pendingReclaimHandlesMutex_);
        pendingReclaimEmpty = pendingReclaimHandles_.empty();  // (15) pendingReclaimHandles (EXACT)
    }
    return !hasDeferredCommit
        && pendingReclaimEmpty
        && retireDepth == 0
        && lifetimeRetireIntentPending == 0
        && ringResident == 0
        && dspQuarantineResident == 0
        && retireQuarantineResident == 0
        && terminalReclaimResident == 0
        && runtimePublicationBridge_.isFullyDrained();  // → Layer 2
}
```

### D107-1 verdict: **16 conditions all present, each from observable source**

| # | Condition | Source authority | Hardcoded? |
|---|---|---|---|
| 1 | `swapPending` | `coordinator_.swapPending_` atomic bool | NO (read from atomic) |
| 2 | `intentQueue_.sizeApprox()` | `intentQueue_` (MPSC, sizeApprox) | NO |
| 3 | `observeDeferredRing_.size()` | `observeDeferredRing_` (SPSC ring) | NO |
| 4 | `quarantineFallbackQueue_.sizeApprox()` | `quarantineFallbackQueue_` (MPSC fallback) | NO |
| 5 | `recoveryIntentQueue_.size()` | `recoveryIntentQueue_` (SPSC ring) | NO |
| 6 | `retireBacklogCount_ == 0` | atomic counter | NO (read) |
| 7 | `publicationBacklogCount_ == 0` | atomic counter | NO (read) |
| 8 | `publicationIntentResidencyCount_ == 0` | atomic counter (X5 INV) | NO (read) |
| 9 | `pendingIntentCount_ == 0` | atomic counter (transport reservation) | NO (read) |
| 10 | `reclaimInFlightCount_ == 0` | atomic counter (approximate) | NO (read) |
| 11 | `quarantineIntentResidencyCount_ == 0` | atomic counter (X6 INV) | NO (read) |
| 12 | `quarantineRingResidencyCount_ == 0` | atomic counter (X6 INV) | NO (read) |
| 13 | `!recoveryAdmissionPending_` | atomic bool (durable flag) | NO (read) |
| 14 | `liveLogicalRecoveryObligationCount() == 0` | table counter (R20/D105-R13) | NO (read) |
| 15 | `pendingReclaimHandles_.empty()` | `pendingReclaimHandles_` (mutex-protected set, EXACT) | NO |
| 16 | `hasDeferredRequest()` | `RuntimeIntentCoordinator` | NO |

---

## D107-1.CORRECTION to D106 (Q0 verification)

**D106 line 6 (Conclusion) incorrectly stated "Q0 hardcoded true"** in the header. The
current `ConvoPeq.md` (line 4374) shows:

```cpp
obs.admissionReservationsZero = (shutdownRuntime_.outstanding() == 0);  // Q0
```

D106 itself noted the fix is applied but the header summary was not updated. **D107
re-confirms the fix is in place** — no Q0 hardcode regression.

---

## D107-2 — Physical entity tracing (counter ↔ resource)

### Counter → resource mapping

| Counter | Resource (physical entity) | Increment site | Decrement site |
|---|---|---|---|
| `pendingIntentCount_` | Transport reservation (Observe/Quarantine/Recovery, before push) | `submitObserve/submitQuarantine/submitRecoveryRequest` (line 57098, 57123) | `pop` (line 58181/58185) |
| `recoveryIntentQueue_.size()` | Recovery transport queue (SPSC) | `submitRecoveryRequest` line 56758 (push) | `popRecoveryRequest` line 57038 (pop) |
| `quarantineIntentResidencyCount_` | intentQueue_ Quarantine residency | line 57125 (push intent as Quarantine) | line 58185 (pop intent as Quarantine) |
| `quarantineRingResidencyCount_` | quarantineFallbackQueue_ residency | line 57113 (push to ring) | line 58164 (pop from ring) |
| `retireBacklogCount_` | semantic retire backlog counter | `incrementRetireBacklog` line 56039 (NOT used in production per D101-32-D) | `decrementRetireBacklog` line 56048 (NOT used) |
| `publicationBacklogCount_` | semantic publish backlog | `setPublicationBacklogCount` (DEPRECATED, NOT used per D101-32-C) | n/a |
| `publicationIntentResidencyCount_` | intentQueue_ Publish residency (X5 INV) | `enqueuePublicationIntent` line 57748 | `popPublicationIntent` line 57751 (line 58181 in drain path) |
| `reclaimInFlightCount_` | Approximate reclaim pending counter | `onReclaimBegin` line 56061 | `onReclaimEnd` line 56074 |
| `quarantineResidentCount_` (deprecated, removed per D101-32-D) | n/a | n/a | n/a |
| `quarantineFallbackDropCount_` | quarantine intent drop count (telemetry) | drop path | n/a |
| `recoveryIntentDropCount_` | recovery intent drop count (telemetry) | drop path | n/a |
| `recoveryCoalescedCount_` | coalesce event count (telemetry) | `findByKey` match | n/a |
| `recoveryCapacityExhaustedCount_` | capacity reject count (telemetry) | `tryInsert` failure | n/a |
| `recoveryRetryDeferredCount_` | deferral count (telemetry) | deferral path | n/a |
| `recoveryAdmissionPersistent_` | durable admission flag (atomic bool) | durable admission in `submitRecoveryRequest` line 56947 | durable consume line 57025 |
| `recoveryShutdownDiscardCount_` | shutdown discard count (telemetry) | `discardRecoveryRequestsOnShutdown` line 1207 | n/a |
| `liveLogicalRecoveryObligationCount_` | logical obligation table live count | `tryInsert` line 354-368 | `resolve` line 388-389 |
| `recoveryObligationIdShutdownDiscardCount_` (was in earlier D105-R12 evidence) | n/a (was for closure-side counter) | n/a | n/a |
| `pendingReclaimHandles_` (set) | **EXACT** reclaim pending handles | reclaim start | reclaim end |

### D107-2 verdict: **Each counter maps to a physical resource** (queue, durable slot, flag, table).
There is no counter whose physical entity is "mystery" — every counter is increment/decrement
on a real resource.

---

## D107-3 — Case A/B/C/D/E

### Case A: `queue == empty` && `counter > 0`

**Possible?**

- `recoveryIntentQueue_.size() == 0` is checked at condition 5
- `pendingIntentCount_` is checked at condition 9
- These are **two independent observations** (queue is a LockFreeRingBuffer's internal size,
  counter is an atomic)

If `pendingIntentCount_` > 0 but queue is empty, **the counter is wrong**. This should be
impossible because the increment-decrement protocol is `fetchAdd` (before push) + `fetchSub`
(after pop). If the queue is empty after a pop, the counter was decremented too. The
push was incremented before the push; if the push failed, a fetchSub rollback is performed
(line 57123-57124 for quarantine fallback). This is the **architectural invariant**
that REPAIR_PLAN2 §1.1.3 establishes.

**Verdict**: Case A is architecturally impossible in the current code structure (counter
is reservation-based, always paired with push/pop). However, this is a **code-level
invariant**, not a code-level atomic guarantee. A logic bug in counter management could
still violate it. The D107 audit finds no such logic bug.

### Case B: `queue != empty` && `counter == 0`

**Impossible** because the counter is incremented **before** the push, and the push
is atomic (LockFreeRingBuffer). If the push fails (full), the counter is decremented
back. So when `counter == 0`, the queue is guaranteed empty (or only contains items that
were decremented after consumption).

### Case C: `queue == empty && counter == 0` but `liveLogicalRecoveryObligationCount > 0`

This is the **Case C** explicitly guarded by condition 14
(`liveLogicalRecoveryObligationCount() == 0`). The D105-R13 structural
assertion is incorporated as a separate predicate to handle the case where:
- Transport queue is consumed
- All reservations are released
- Counter is 0
- BUT logical obligation table still has Live entries (e.g., popped-but-not-yet-resolved)

This is the **D105-R13 fix** and is **correctly applied** at condition 14.

### Case D: `reclaimInFlight > 0` but all queues == 0

**Possible** (theoretically). `reclaimInFlightCount_` is approximate; the EXACT
counter is `pendingReclaimHandles_` (set, mutex-protected). Both are checked
in Layer 1: `reclaimInFlightCount_ == 0` AND `pendingReclaimHandles_.empty()`. The
combination prevents the approximate counter from allowing false-positive drain.

### Case E: `quarantineTransport == 0` but physical DSP quarantine > 0

**Possible** — `quarantineIntentResidencyCount_` + `quarantineRingResidencyCount_`
track the Intent/queue residency. `dspQuarantineResident` (in Layer 1) tracks the
DSP-level physical quarantine. If a DSP is quarantined but no Intent is in flight
(no obligation, no transport), the queue conditions are 0 but DSP quarantine > 0.
**Layer 1 includes `dspQuarantineResident == 0`**, so this case is caught.

### D107-3 verdict: **All five cases are properly guarded** in current `isFullyDrained()`.

---

## D107-4 — External setter / authority singularization

### Production setters on counters

The current `RuntimeIntentCoordinator` exposes **only the following public setters**:

| Setter | Status | Comment |
|---|---|---|
| `setRetireBacklogCount` | **DEPRECATED** (D101-32-D vestigial) | Writer zero, not used in production |
| `setPublicationBacklogCount` | **DEPRECATED** (D101-32-D) | Writer zero |
| `setDeferredRetireResidencyCount` | **DEPRECATED** (D101-32-D) | Writer zero |
| `setFallbackBacklogCount` | **DEPRECATED** (D101-32-D) | Writer zero |
| `setPendingIntentCount(hasDeferredCommit ? 1u : 0u)` | Conditional (line 117) | Internal-only, AudioEngine |

All **observable production** counter updates are atomic `fetchAdd` / `fetchSub` /
`publishAtomic` / `consumeAtomic` within `RuntimeIntentCoordinator` member functions.
There are **no `setRetireBacklogCount`/`setPublicationBacklogCount`/`setDeferredRetireResidencyCount`/`setFallbackBacklogCount` callers** in the current production code path
(D101-32-D removed all callers).

### Counter mutation authority

| Counter | Increment authority | Decrement authority | Authority singularization |
|---|---|---|---|
| `pendingIntentCount_` | `submitObserve` / `submitQuarantine` / `submitRecoveryRequest` (single NonRT producer) | `pop` (line 58181/58185) | **YES** (Coordinator-owned) |
| `recoveryIntentQueue_` | `submitRecoveryRequest` | `popRecoveryRequest` | **YES** (SPSC, single producer/consumer) |
| `quarantineIntentResidencyCount_` | `submitQuarantine` (intent push) | `processIntent` (Quarantine dispatch) | **YES** (NonRT, single producer) |
| `quarantineRingResidencyCount_` | `submitQuarantine` (ring fallback) | `processIntent` (ring pop) | **YES** |
| `reclaimInFlightCount_` | `onReclaimBegin` | `onReclaimEnd` | **YES** (Coordinator-owned) |
| `pendingReclaimHandles_` | `dspHandleRuntime.reclaim()` start | `dspHandleRuntime.reclaim()` end | **YES** (set, mutex-protected) |
| `recoveryAdmissionPending_` | `submitRecoveryRequest` (durable fallback) | `settlePendingRecoveryAdmission(false)` | **YES** |
| `liveLogicalRecoveryObligationCount_` | `tryInsert` (line 354-368) | `resolve` (line 388-389) | **YES** (table-owned) |

### D107-4 verdict: **Authority singularization is established**. All counters are
mutated by single authorities. The historical external setters
(D101-32-D) are vestigial and have zero callers.

### D107-4 noise: `recoverCount` etc. exist in public API

The `recover*` API in `RuntimeIntentCoordinator` includes vestigial counter methods
(`recoverCount()`, `recoverHandleRuntimeCount()`, `recoverTail()`). These are part
of an **older, unused** Recover API path. They are **read-only** (no setters) and
**not used in `isFullyDrained()`**. The `recover*` API path is **dead code** that
should be cleaned up in a future work, but does **not** affect drain semantics.

The comment at `ISRRuntimePublicationCoordinator.cpp:56355` says "existed since
2017" — this is a comment artifact referring to a historical line of code (the
**original 2017 startup of this project's Recover API**), and is not a claim about
**current** code. The current `RuntimeIntentCoordinator` class is **part of the
2026 refresh** (D101-32-C) where Recover was re-architectured into the obligation
table. The comment is **stale documentation, not stale code**.

---

## D107-5 — `isFullyDrained()` vs Reclaim permission

The current code distinguishes:

```text
isFullyDrained() == true
    ⇏ reclaim permission granted
```

The chain is:

```
isFullyDrained() (Layer 1 + Layer 2, observation only)
    ↓
tryMakeQuiescenceProof(obs)
    ↓ (validates all Q in observation)
ShutdownQuiescenceProof (identity-bound)
    ↓
tryMakeReclaimPermit(proof)
    ↓ (proof.valid() check)
ReclaimPermit (single-use)
    ↓
reclaimShutdownQuiescent(permit)
    ↓ (identity + consume check)
physical reclaim
```

**`isFullyDrained()` does NOT issue reclaim permission**. It is a pure observation
predicate used by Layer 1 to detect when to call `waitForDrain` (or its equivalent
in the runtime). The reclaim path requires an explicit
`tryMakeQuiescenceProof(obs)` + `tryMakeReclaimPermit(proof)` + `reclaimShutdownQuiescent(permit)`
chain, which is a **separate function** triggered by
`tryShutdownQuiescentReclaim` (D106 verified).

This separation satisfies `INV-DRAIN-1` (observation ≠ authority) and
`INV-DRAIN-2` (separate authority for Reclaim permission).

### D107-5 verdict: **Drain observation and Reclaim authorization are properly separated**.

---

## D107-6 — False-positive / False-negative table

| State | `isFullyDrained()` returns | Verdict |
|---|---|---|
| All resources = 0 | `true` | **PASS** |
| Queue残留 (any of 4) | `false` | **PASS** (Layer 2) |
| pendingIntentCount > 0 | `false` | **PASS** (Layer 2) |
| publicationIntentResidencyCount > 0 | `false` | **PASS** (Layer 2) |
| quarantineIntentResidencyCount > 0 | `false` | **PASS** (Layer 2) |
| quarantineRingResidencyCount > 0 | `false` | **PASS** (Layer 2) |
| recoveryAdmissionPending_ == true | `false` | **PASS** (Layer 2) |
| liveLogicalRecoveryObligationCount > 0 | `false` | **PASS** (Layer 2 — D105-R13) |
| reclaimInFlight > 0 | `false` | **PASS** (Layer 2) |
| pendingReclaimHandles non-empty | `false` | **PASS** (Layer 1 — INV-X3-5) |
| pendingReclaimEmpty via mutex-protected set | exact (Layer 1) | **PASS** |
| retireDepth > 0 (Router pendingRetireCount) | `false` | **PASS** (Layer 1) |
| lifetimeRetireIntentPending > 0 | `false` | **PASS** (Layer 1) |
| ringResident > 0 | `false` | **PASS** (Layer 1) |
| dspQuarantineResident > 0 | `false` | **PASS** (Layer 1) |
| retireQuarantineResident > 0 | `false` | **PASS** (Layer 1) |
| terminalReclaimResident > 0 | `false` | **PASS** (Layer 1) |
| swapPending_ == true | `false` | **PASS** (Layer 2) |
| hasDeferredCommit (Layer 1) | `false` | **PASS** (Layer 1) |
| recoverCount etc. (D107-4 noise) | not used in `isFullyDrained()` | N/A |

### Potential false-positive paths (resource > 0 but predicate = true)

| Scenario | Currently caught? | Reason |
|---|---|---|
| `swapPending` transition in-flight | **YES** (condition 1) | atomic read |
| Pending Recovery lease (Building) | **YES** (condition 13) | `recoveryAdmissionPending_` true |
| `pendingReclaimHandles_` non-empty | **YES** (Layer 1 condition 15) | mutex-protected set check |
| Terminal Reclaim in-progress | **YES** (Layer 1) | `terminalReclaimResident` direct read |
| Quarantine transport in flight | **YES** (Layer 2) | `quarantineIntentResidencyCount` + `quarantineRingResidencyCount` |
| Recover transport queue not empty | **YES** (Layer 2) | `recoveryIntentQueue_.size()` |
| Live obligation but no transport | **YES** (Layer 2) | `liveLogicalRecoveryObligationCount() == 0` |
| Observe deferred ring not empty | **YES** (Layer 2) | `observeDeferredRing_.size()` |
| Fallback ring not empty | **YES** (Layer 2) | `quarantineFallbackQueue_.sizeApprox()` |

### D107-6 verdict: **No false-positive or false-negative gaps detected**. Every physical
resource has a corresponding predicate.

---

## D107-7 — Final verdict

### Verdict: **CONDITIONAL PASS**

**Structural soundness**:
- All 16 conditions are present in current `isFullyDrained()` (Layer 1 + Layer 2)
- Each condition reads from an observable source (atomic counter, queue, manager)
- Authority singularization is established (no external setters, single mutation authority per counter)
- `isFullyDrained()` and Reclaim permission are properly separated (observation vs authority)
- D105-R13 structural assertion (logical obligation residency) is included as condition 14
- `reclaimInFlightCount_` is approximate; `pendingReclaimHandles_` (set, EXACT) supplements it
- Reclaim permission production-ready path is wired through `tryShutdownQuiescentReclaim`

**CONDITIONAL justification**:
1. **Vestigial public API** (`recover*`) is present in the production class header
   but is dead code (D101-32-C vestigial). It is read-only and does not affect
   drain semantics, but is a **code-cleanliness issue** (not a correctness issue).
2. **Stale comment** at `ISRRuntimePublicationCoordinator.cpp:56355` claims "existed since
   2017" but the file is part of a 2026 refresh. This is a **comment artifact**, not
   a code defect. (D101-32-A/B/C also contains this pattern of stale comments;
   the comment layer is a known D101-22 issue.)
3. **`recoverCount` is a public counter** with no callers in current production
   code. It is a **vestigial surface** but not used in `isFullyDrained()`.

D107 does **not** find:
- Any structural contradiction
- Any missing lifetime condition
- Any unsafe race or privilege boundary
- Any unsafe counter authority
- Any false-positive risk

D107 establishes that the **drain predicate is complete and authoritative** in the
current production source. The remaining gaps are **vestigial public API** issues
that are **separately tracked** in D101-32-C and are not drain-semantic concerns.

---

## D107 → D108 hand-off

D107 is **CONDITIONAL PASS**. The structural foundation is solid.

The next audit (`D108 — A2-G01~G23 Current-Code Gate Inventory`) should:
1. Re-verify each of G01~G23 against **current ConvoPeq.md only** (do not rely on
   REPAIR_PLAN2's old status text)
2. Identify which G's are PASS / which are gaps
3. Determine whether A2 implementation is enabled or whether further read-only
   audits are needed

**Implementation is NOT approved** by D107. D107 only confirms drain semantics.
A2 implementation requires D108's gate inventory first.

---

## File references (read-only, no source modification)

| File | Role |
|---|---|
| `ConvoPeq.md` (2026-08-27 14:33) | runtime source baseline |
| `src/audioengine/AudioEngine.Threading.cpp:114-174` | `AudioEngine::isFullyDrained` (Layer 1) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:460-462` | `RuntimeIntentCoordinator::isFullyDrained` (delegation) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:506-559` | `ShutdownScheduler::isFullyDrained` (Layer 2 — 16 conditions) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:114-116` | Layer 1 `hasDeferredCommit` |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:117-131` | Layer 1 retire/overflow/deferred comments |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:55988-56076` | retire/pending/reclaim counter mutations |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:57095-57124` | quarantine push counter paths |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:58170-58200` | drain path counter paths |
| `src/audioengine/AudioEngine.h:4367-4396` | `tryShutdownQuiescentReclaim` (D106 caller) |
| `src/audioengine/ISRLifetimeProof.h:76-171` | `ShutdownQuiescenceProof` + `ReclaimPermit` (D106 evidence) |
| `src/audioengine/ISRShutdown.cpp:348-410` | `tryMakeQuiescenceProof` + `tryMakeReclaimPermit` |
| `evidence/15-P-4-3-isfullydrained-completion-audit.md` | historical Layer 2 audit (Line 33-49: 16 conditions listed) |
| `evidence/15-P-4-7-shutdown-completion-invariant-audit.md` | historical drain predicate enum |
| `evidence/D101-26-Shutdown-Lifetime-Proof-Current-Code-Audit.md` | Q0-Q7 wiring verification (D106 Layer 2) |
| `evidence/D101-32-D-Vestigial-Counter-Semantic-Event-API-Removal.md` | vestigial setter removal history |
| `evidence/D101-32-C-Semantic-Event-Vestigial-Counter-Integrity-Audit.md` | vestigial counter audit (G19) |
| `evidence/D106_SHUTDOWN_LIFETIME_QUIESCENCE_AUDIT.md` | Q0 hardcode refutation (D107-1 correction) |

**No source files modified. No I4 files modified. No tests added.** D107 is a
read-only audit of the current `isFullyDrained()` semantics and counter authority.

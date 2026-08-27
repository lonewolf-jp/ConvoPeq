# D105-R5-8 — Recovery Logical Obligation Enforcement Implementation Plan

**Type**: Read-only implementation-plan / audit (defines boundaries; **0 source changes in this step**)
**Purpose**: Concretely fix the implementation boundaries for enforcing `liveLogicalRecoveryObligationCount ≤ 32`, based on the proven R5-rev1 / R6 / R7 / R5-7 model. Actual code mutation happens in the subsequent **Implementation** phase, verified by **D105-R8**.
**Date**: 2026-08-27
**Status**: ✅ Plan fixed (R5-7-GATE = PASS ⇒ implementation now permitted)
**Prereq**: D105-R5 (R5-rev1), D105-R6, D105-R7, D105-R5-7

---

## 0. Scope & Non-Goals

**In scope (this plan)**: types, data structures, function signatures, authority boundaries, modification points, telemetry, ordering, verification entry points.
**Out of scope (separate Implementation phase)**: actual edits; **D105-R8** verifies them.
**Explicitly NOT changed** (per R5 §10 #12, R5-7 §9):
- `kMaxSlots = 256` (ISRDSPQuarantine.h:68) — trigger source, not the bound.
- `recoveryIntentQueue_` capacity 256 (h:642) — delivery buffer.
- `pendingIntentCount_` — keep for `isFullyDrained` only.
- Do **NOT** add a bare `kMaxLogicalRecoveryObligations = 32` constant *without* the table/accounting (INV-CAP-7 requires the bound to be *enforced*, not just declared).

---

## 1. New Types

### 1.1 `LogicalRecoveryObligationId` (R7 §1/§2)
```cpp
using LogicalRecoveryObligationId = uint64_t;
// allocated by Coordinator only:
std::atomic<LogicalRecoveryObligationId> nextLogicalRecoveryObligationId_{0};
// monotonic; 64-bit ⇒ no practical wraparound; table cleared on shutdown ⇒ safe reuse
```

### 1.2 `SemanticRecoveryTarget` (R5 §3.1 — seed already exists)
```cpp
struct SemanticRecoveryTarget {
    uint64_t irIdentityHash;
    uint64_t convolutionConfigHash;
    uint64_t dspParameterHash;
    uint64_t convolverFingerprint;
    uint64_t sampleRate;
    // equality from buildSource.rebuildFingerprint (already populated:
    //   AudioEngine.RebuildDispatch.cpp:95-100; equality exists RuntimeBuildTypes.h:315-336)
};
```

### 1.3 `CoalesceIdentity` (R5 §3.2)
```cpp
struct CoalesceIdentity {
    DSPHandle quarantinedHandle;      // excluded DSP
    SemanticRecoveryTarget target;      // resulting-world config
    bool operator==(const CoalesceIdentity&) const; // handle== && target==
};
```

### 1.4 `LogicalRecoveryObligation` (R5 §3 / R7)
```cpp
enum class ObligationState : uint8_t {
    Created, Transport, Durable, Building,
    Completed, Failed, StaleSuperseded, Superseded, ShutdownDiscard
};
struct LogicalRecoveryObligation {
    LogicalRecoveryObligationId id;
    CoalesceIdentity            coalesceKey;   // locked at creation (Option 1)
    ObligationState             state;
    convo::RuntimeBuildSnapshot buildSource;    // O owns original (R7 §3/§4)
    uint64_t                    generation;
};
```

### 1.5 `RecoveryResolution` (R7 §5)
```cpp
enum class RecoveryOutcome { Published, Retry, Failed, StaleSuperseded, ShutdownDiscard };
struct RecoveryResolution { LogicalRecoveryObligationId id; RecoveryOutcome outcome; };
```

---

## 2. `RecoveryAdmissionTable<32>` (R5 §10 #1, R5-7 §6)

```cpp
// fixed-capacity, Coordinator-owned; the ENFORCED resource bound (INV-CAP-7)
static constexpr size_t kMaxLogicalRecoveryObligations = 32;
struct RecoveryAdmissionTable {
    LogicalRecoveryObligation slots[kMaxLogicalRecoveryObligations];
    // lookup by CoalesceIdentity (for coalesce check, O(32) linear scan acceptable)
    LogicalRecoveryObligation* findByKey(const CoalesceIdentity&);
    // lookup by id (for resolution)
    LogicalRecoveryObligation* findById(LogicalRecoveryObligationId);
    // insert returns nullptr if full
    LogicalRecoveryObligation* tryInsert(LogicalRecoveryObligation&&);
    size_t liveCount() const;   // = liveLogicalRecoveryObligationCount equivalent
};
std::atomic<uint64_t> liveLogicalRecoveryObligationCount_{0};  // mirror for O(1) checks
```
**Note**: `liveLogicalRecoveryObligationCount_` is the *enforced* counter; table scan `liveCount()` is authoritative for correctness, atomic for the fast admission guard.

---

## 3. Carry-Field Additions (R7 §2/§6/§13)

| Struct | Field | Init point |
|---|---|---|
| `RecoveryIntent` (h:217) | `LogicalRecoveryObligationId obligationId{0};` | set at admission (R5 §10 #7) |
| `PendingRecoveryAdmission` (h:667) | `LogicalRecoveryObligationId obligationId{0};` | copied from intent |
| `PublishRequest` (PublicationAdmission.h:19) | `uint64_t recoveryObligationId{0};` | Builder → publish (R7 §6) |

---

## 4. Admission Flow Modification — `submitRecoveryRequest` (R5 §4, R5-7 §6)

Current: cpp:812-875 (push → on full blind durable overwrite = **INV-X1-7 defect**).
New contract:
```
submitRecoveryRequest(handle, buildSource, epoch):
    C = CoalesceIdentity{handle, deriveTarget(buildSource)}
    // 1. COALESCE (before full-check)
    if (auto* O = table.findByKey(C)) {
        record recoveryAdmissionCoalescedCount_++;
        return true;                       // ΔL = 0, no new transport entry
    }
    // 2. CAPACITY GUARD (unique admission only)
    if (liveLogicalRecoveryObligationCount_ == 32) {
        record recoveryAdmissionRejectedCount_++;   // INV-5: observable, not silent
        return false;                      // REJECT, no Created, L stays 32
    }
    // 3. UNIQUE ADMISSION (+1)
    id = nextLogicalRecoveryObligationId_++;
    O = table.tryInsert({id, C, Created, buildSource, generation});
    liveLogicalRecoveryObligationCount_++;            // +1 (guarded by step 2)
    // 4. DELIVERY (ΔL=0)
    intent.obligationId = id;
    if (!recoveryIntentQueue_.push(intent)) {        // queue full
        pendingRecoveryAdmission_ = durableFrom(intent);   // durable slot (single, P3)
    }
    return true;
```
**Removes** the blind overwrite (INV-X1-7 fix). Coalesce precedes the full-check (R5-7 §6 Case C).

---

## 5. Completion Flow — Builder → Coordinator (R5-rev1 §6, R7 §5/§6)

### 5.1 Builder emits resolution (AudioEngine.RebuildDispatch.cpp, recovery loop ~911-1074)
```
RecoveryIntent intent = popped/taken;     // carries obligationId
if (build failed) {
    if (has durable reservation)  emit{intent.obligationId, Retry};     // → Durable (ΔL=0)
    else                         emit{intent.obligationId, Failed};     // → −1
} else {
    recoverySnapshot = intent.buildSource; recoverySnapshot.generation = rebuildRequestGeneration;
    enqueuePublicationIntentForRuntimeCommit(dsp, generation, recoverySnapshot, intent.obligationId);
    // outcome decided at onPublishCommitted, NOT here
}
```
### 5.2 `onPublishCommitted` carries id (RuntimePublicationOrchestrator.h:146 / cpp:329)
```
void onPublishCommitted(PublicationSequenceId seqId,
                        uint64_t recoveryObligationId = 0);   // 0 = normal (non-recovery) publish
// only when recoveryObligationId != 0 → route to completion authority:
if (recoveryObligationId) completionAuthority.resolve({recoveryObligationId, Published});
```
### 5.3 `enqueuePublicationIntentForRuntimeCommit` (AudioEngine.h:2551 / Commit.cpp:782) — add `uint64_t recoveryObligationId` param, store into `PublishRequest.recoveryObligationId`, propagate through `submitPublishRequest` → executor intent → `onPublishCommitted`.
### 5.4 Completion authority (single −1) (R7 §10)
```
void resolve(RecoveryResolution r):
    O = table.findById(r.id);
    if (!O || O->state terminal) return;          // idempotency guard
    if (r.outcome == Retry) { O->state = Durable; return; }   // ΔL=0
    // terminal:
    O->state = terminalOf(r.outcome);
    liveLogicalRecoveryObligationCount_--;        // exactly one −1
    table.remove(O);
```
Publish `FailureReason` classification (RuntimePublicationOrchestrator.cpp:357-386) → outcome:
`QueuePressure`→Retry; `StaleGeneration`→StaleSuperseded; `ValidationFailed`/`PublishFailed`→Failed; `ShutdownRejected`→ShutdownDiscard (R7 §8).

---

## 6. Shutdown Flow (R6 §8, R7 §9)

`requestShutdown()` (or `discardRecoveryRequestsOnShutdown` / `discardPendingRecoveryAdmission`):
```
for (auto& O : table) if (O live) resolve({O.id, ShutdownDiscard});   // −1 exactly once
// side-effect drains (do NOT emit per-entry −1):
discardRecoveryRequestsOnShutdown();   // transport queue
discardPendingRecoveryAdmission();     // durable slot
builder.stop();
```
Channel drains remain delivery cleanup only (P2/P3). Existing `recoveryShutdownDiscardCount_` records the table-iteration terminal count.

---

## 7. Telemetry (INV-5 — observable, never silent)
- `recoveryAdmissionCoalescedCount_` (new)
- `recoveryAdmissionRejectedCount_` (new — replaces silent reject at L==32)
- `recoveryShutdownDiscardCount_` (exists; now per-obligation)
- keep `recoveryIntentDropCount_`, `pendingIntentCount_` (delivery only)

---

## 8. Source Modification Map

| File | Symbol | Change |
|---|---|---|
| ISRRuntimePublicationCoordinator.h | `RecoveryIntent` | +`obligationId` |
| ISRRuntimePublicationCoordinator.h | `PendingRecoveryAdmission` | +`obligationId` |
| ISRRuntimePublicationCoordinator.h | (new) `LogicalRecoveryObligation`, `CoalesceIdentity`, `SemanticRecoveryTarget`, `RecoveryResolution`, `RecoveryAdmissionTable<32>`, `nextLogicalRecoveryObligationId_`, `liveLogicalRecoveryObligationCount_`, telemetry | add |
| ISRRuntimePublicationCoordinator.cpp | `submitRecoveryRequest` | coalesce-then-guard-then-admit (§4); remove blind overwrite |
| ISRRuntimePublicationCoordinator.cpp | `popRecoveryRequest` / `takePendingRecoveryAdmission` | carry `obligationId` |
| ISRRuntimePublicationCoordinator.cpp | `settlePendingRecoveryAdmission` | unchanged (delivery only) |
| ISRRuntimePublicationCoordinator.cpp | `discardRecoveryRequestsOnShutdown` / `discardPendingRecoveryAdmission` | unchanged (drain); table-iter added in shutdown |
| PublicationAdmission.h | `PublishRequest` | +`recoveryObligationId` |
| RuntimePublicationOrchestrator.h/.cpp | `onPublishCommitted` | +`recoveryObligationId` param |
| RuntimePublicationOrchestrator.cpp | `submitPublishRequest` | propagate `recoveryObligationId`; classify `FailureReason`→outcome |
| AudioEngine.h / AudioEngine.Commit.cpp | `enqueuePublicationIntentForRuntimeCommit` | +`recoveryObligationId` param; store into `PublishRequest` |
| AudioEngine.RebuildDispatch.cpp | recovery build loop | emit `RecoveryResolution` for build outcomes; pass `obligationId` to publish |

---

## 9. Ordering / Phasing (for Implementation phase)
1. Add types + `RecoveryAdmissionTable<32>` + `liveLogicalRecoveryObligationCount_` (§1–§2).
2. Add carry fields (§3).
3. Rewrite `submitRecoveryRequest` admission (§4) — **fixes INV-X1-7**.
4. Plumb `obligationId` through publish → `onPublishCommitted` (§5.2–5.3).
5. Add Builder resolution emission (§5.1) + completion authority (§5.4).
6. Shutdown table-iteration (§6).
7. Telemetry (§7).
8. **Do NOT** change queue 256 / kMaxSlots 256 / `pendingIntentCount_`.

---

## 10. Verification Entry Point — D105-R8
Post-implementation structural verification must re-confirm:
- exactly one `+1` site (admission, post-coalesce, guarded by L<32);
- exactly one `−1` authority (completion, idempotent);
- `reclaimSlot` still does not touch L;
- `PendingRecoveryAdmission` still single slot;
- queue 256 / kMaxSlots 256 untouched;
- `pendingIntentCount_` unchanged in semantics;
- all R7 §11 states reachable; no new transition outside R7 §11;
- stress/soak: 256 concurrent quarantines ⇒ L never exceeds 32 (coalesce + reject).

---

## 11. Traceability
- R5 §10 #1–#7, #10–#12 (minimal design changes → concrete boundaries here).
- R6 §11 (required R5-rev1 revision → §4/§5 here).
- R7 §2/§5/§6/§8/§9 (protocol → carry fields, completion, shutdown).
- R5-7 §12 (proof lemmas → enforced by §4/§5/§6).

### Tools used (cross-validation of plan against code)
- WSL rg/sed/rtk, context-mode MCP (this & prior audits)
- AiDex, semble, cocoindex (INV-CAP-7 D25), graphify (prior audits)
- serena MCP (timed out; corroborated by others)

**No source modified in D105-R5-8.**

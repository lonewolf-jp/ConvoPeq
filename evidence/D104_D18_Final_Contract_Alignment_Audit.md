# D104 — I4 D18 Final Contract Alignment Audit

**Type**: Read-only audit (D103 → D18 realignment)
**Constraint**: production source = 0, test source = 0, CMake = 0, contract = 0
**Status**: ⚠️ CONDITIONAL GO (3 blocking items remain: E, F, backpressure proof)
**Base**: D103 audit findings, realigned against I4_DESIGN_CONTRACT.md D18.1–D18.6, D19–D22
**Date**: 2026-08-15 (Design-8 consolidation)

---

## Executive Summary

D103 concluded: "All supersession-related identifiers are design-only (0 production implementations). NO-GO."

**D104 realignment verdict**: D103's conclusion was directionally correct but structurally misaligned with D18. The core problem is NOT missing `canSupersede()` — it is that **Phase I's entire supersession/supersede path is structurally inactive by design** (D18.2). The actual Phase I focus has shifted from "supersession readiness" to **coalesce infrastructure + domain identity separation + ownership conservation completeness**.

**Revised Phase I readiness**: COALESCE infrastructure missing + 3-domain separation missing + conservation equation incomplete (missing `successCount`) + `buildInputHash` field missing from semantic target struct.

### D103 → D104 Realignment Summary

| D103 Focus | D104 (D18-corrected) Focus |
|---|---|
| "canSupersede() missing" | "coalesce infrastructure missing (canSupersede is Phase II extension point)" |
| "supersession gate" | "coalesce gate (identical target = COALESCE, not supersede)" |
| "isSemanticTargetSuperset missing" | "isSemanticTargetSuperset IS Phase I (equality-based), but SemanticRecoveryTarget struct missing buildInputHash" |
| "RecoveryEpisodeId missing" | "RecoveryEpisodeId missing — MUST be implemented as identity lineage (D13)" |
| "ownership conservation broken (D15)" | "ownership conservation equation missing successCount (D18.3 fix)" |

---

## 1. Audit 0 — D18.1 Re-alignment (Coalesce vs Supersede Boundary)

### Finding: Phase I is COALESCE-centered, NOT SUPERSEDE-centered

**D103 Audit conclusion**: Phase I lacks `canSupersede()` predicate.
**D18.1 correction**: `canSupersede()` is structurally **INACTIVE in Phase I**. The boundary is:

```
Step 0: CoalesceIdentity 検索
    CoalesceIdentity = { handle, RecoveryEpisodeId, SemanticRecoveryTarget }   ← RecoveryGeneration を含まない
    existing が同一 CoalesceIdentity → COALESCE
    (identical target → coalesce, NOT supersede)

Step 1: Supersede decision (Phase II extension point)
    canSupersede(newer, older) — only meaningful when targets DIFFER
    (Phase I: equality-based isSemanticTargetSuperset — identical targets coalesce before reaching supersede)
```

### D103 Audit 0 Re-assessment

The D103 Audit 0 searched for `canSupersede` (0 hits) and correctly reported it as MISSING. However, per D18.1:

- **`canSupersede()` is NOT required for Phase I implementation** — it is an extension point for Phase II (when targets differ and partial-order containment is needed).
- **`isSemanticTargetSuperset()` IS required for Phase I** — but with equality semantics (D18.2: `identical target = all 5 hash fields equal`).
- The D103 audit correctly identified that `isSemanticTargetSuperset` has 0 hits, but the fix is **NOT** to implement a full superset predicate — it is to implement `SemanticRecoveryTarget` with all 5 fields (including `buildInputHash`) and use **equality** as the Phase I semantic containment check.

### Source Code Verification (Read-Only)

- ✅ `ISRRuntimePublicationCoordinator.cpp:865-872`: `submitRecoveryRequest` performs **blind overwrite** on queue-full — NO coalesce identity search, NO `LogicalRecoveryIdentity` comparison. This is the D103 finding and is **CONFIRMED**.
- ✅ `ISRRuntimePublicationCoordinator.h:654-681`: `PendingRecoveryAdmission` is a **single-slot struct** — confirmed. No `CoalesceIdentity` lookup.
- ✅ `RuntimeBuildTypes.h:301-337`: `isRuntimeBuildSnapshotSealedAndCompatible()` performs field-by-field **equality** (not superset). Closest to Phase I semantic, but operates on wrong abstraction level.

### Re-aligned Verdict

| Item | D103 Finding | D18.1 Status | Phase I Requirement |
|---|---|---|---|
| `canSupersede()` | 0 hits, MISSING | INACTIVE (Phase II) | NOT required for Phase I |
| CoalesceIdentity search | MISSING | ✅ Required (D18.1 Step 0) | MUST implement |
| `LogicalRecoveryIdentity` | MISSING | ✅ Required (R1) | MUST implement |
| `isSemanticTargetSuperset()` | 0 hits, MISSING | ✅ Phase I (equality) | MUST implement as equality |

---

## 2. Audit 1 — D18.2 Re-alignment (Phase I Semantic Supersession Structure)

### Finding: Phase I supersession is semantically a subset of coalesce

**D18.2**: Phase I `canSupersede()` is structurally a special case:
```
canSupersede(newer, older) =
    same handle
  + same RecoveryEpisodeId
  + isAfter(newer.recoveryGeneration, older.recoveryGeneration)
  + semanticTargetEqual(newer, older)   ← CoalesceIdentity equality
```

This means: **if targets are equal → coalesce (Step 0 catches it). If targets differ → NOT supersede in Phase I (equality predicate says false).**

### D103 Audit 1 Re-assessment

The D103 Audit 1 examined the `SemanticRecoveryTarget` 5-field structure. With D18.2:

- **Phase I semantic containment = exact equality** of all 5 fields (D9.3 truth table: identical → supersede = coalesce).
- The D103 finding that `isSemanticSuperset` truth table includes partial-order cases ({IR+EQ} ⊇ {IR}) is **correct for Phase II** but **Phase I uses equality only** (D18.2: `semanticTargetEqual` not `semanticTargetSuperset`).
- D18.2 explicitly states: "compositional superset を Phase I では採用しない" — this resolves the D9.3 ambiguity in favor of **equality**.

### SemanticRecoveryTarget Field Status (Re-verified)

| Field | Source Code Availability | Status |
|---|---|---|
| `irIdentityHash` | `RuntimeBuildFingerprint.irIdentityHash` (h:41) | ✅ Available |
| `convolutionConfigHash` | `RuntimeBuildFingerprint.convolutionConfigHash` (h:42) | ✅ Available |
| `convolverFingerprint` | `RuntimeBuildSnapshot.convolverFingerprint` (h:52) | ✅ Available |
| `dspParameterHash` | `RuntimeBuildFingerprint.dspParameterHash` (h:43) | ✅ Available |
| **`buildInputHash`** | **0 hits in production source** | ❌ **MISSING — BLOCKING** |

### buildInputHash Gap Analysis

- `isRuntimeBuildSnapshotSealedAndCompatible()` (h:301) performs **individual field comparisons** of `buildInput` struct (sampleRate, blockSize, ditherBitDepth, oversamplingFactor, etc.) — **no hash**.
- `BuildInput` (h:20-34) has 16 fields but is **not hashed** anywhere in production code.
- D18.2 requires `buildInputHash` as a field in `SemanticRecoveryTarget` for Config domain containment.
- **Fix**: Define `canonicalHash(BuildInput)` → `std::uint64_t` and add `buildInputHash` field to `SemanticRecoveryTarget`.

### Verdict

| Item | Status | Phase I Impact |
|---|---|---|
| `SemanticRecoveryTarget` struct | ❌ MISSING | MUST implement (5 fields + equality predicate) |
| `buildInputHash` | ❌ MISSING (0 hits) | **BLOCKING** — required for Config domain equality |
| Equality-based superset | ❌ MISSING | MUST implement as `semanticTargetEqual()` |

---

## 3. Audit 2 — D18.3 Re-alignment (Ownership Conservation with successCount)

### Finding: D15 ownership equation is BROKEN — D18.3 fixes with `successCount`

**D18.3 correction (user指摘4)**: The original D15 equation is structurally broken — `success` transitions an obligation out of live ownership but was not in the equation:

```
❌ D15 (broken):
live = transport + durable + building + stalled + superseded + shutdownDiscard == admittedLogicalObligationCount
  → R1 admitted → Building → Success: left=0, right=1 → BROKEN

✅ D18.3 (fixed):
liveOwnershipCount      = transportCount + durableCount + buildingCount + stalledCount
terminalDispositionCount = successCount + supersededCount + shutdownDiscardCount
admittedLogicalObligationCount = liveOwnershipCount + terminalDispositionCount
```

### D103 Audit 2 Re-assessment

The D103 Audit 2 correctly identified counters:

| Counter | Production Hits | Status |
|---|---|---|
| `pendingIntentCount_` | Multiple (`ISRRuntimePublicationCoordinator.cpp:850,861`) | ✅ transport reservation |
| `recoveryShutdownDiscardCount_` | `h:625` + `cpp:913` | ✅ terminal (ShutdownDiscard) |
| `successCount` | **4 hits, ALL test-only** (`invariant_INV3_INV5.cpp:827,831,833,840`) | ❌ **MISSING (production)** |
| `supersededCount` | **0 hits** | ❌ MISSING |
| `admittedLogicalObligationCount` | **0 hits** | ❌ MISSING |
| `stalledCount` | **0 hits** | ❌ MISSING |
| `recoveryIntentDropCount_` | `h:616` + `cpp:863` | ✅ terminal (drop telemetry, NOT ownership) |

### Source Code Counter Inventory (Re-verified)

```
ISRRuntimePublicationCoordinator.h:
  std::atomic<uint64_t> nextObserveIntentId_{0};                    // :618 — sequence only
  std::atomic<uint64_t> observeOverflowCounter_{0};                // :620 — overflow telemetry
  std::atomic<uint64_t> observeFallbackOverflowCounter_{0};       // :622 — overflow telemetry
  std::atomic<uint64_t> nextRecoveryIntentId_{0};                 // :630 — sequence only (intentId)
  std::atomic<uint64_t> recoveryIntentDropCount_{0};              // :616 — drop telemetry (INV-X1-3)
  std::atomic<uint64_t> recoveryShutdownDiscardCount_{0};         // :625 — terminal (ShutdownDiscard)
  std::atomic<bool> recoveryAdmissionPending_{false};              // :684 — state flag (not counter)
  PendingRecoveryAdmission pendingRecoveryAdmission_;             // :682 — single-slot struct

ISRRuntimePublicationCoordinator.cpp:
  pendingIntentCount_ (fetchAdd/fetchSub managed via ReservationToken)  // cpp:850,861
```

### What's Missing Per D18.3

| D18.3 Counter | Production | Test | Phase I Need |
|---|---|---|---|
| `liveOwnershipCount` (composite) | ✅ derivable from `pendingIntentCount_` + state checks | — | MUST add explicit counter |
| `terminalDispositionCount` (composite) | ❌ | — | MUST add |
| `successCount` | ❌ | ✅ (test-only) | **MUST add to production** |
| `supersededCount` | ❌ | ❌ | ✅ Must be 0 in Phase I (D18.1 inactivity) but equation must include it |
| `admittedLogicalObligationCount` | ❌ | ❌ | MUST add |
| `stalledCount` | ❌ | ❌ | MUST add (D10 — stall ownership) |
| `admissionEventCount` | ❌ | ❌ | MUST add |

### Verdict

| Item | Status | Phase I Impact |
|---|---|---|
| `successCount` | ❌ MISSING (test-only) | **BLOCKING** — D18.3 equation incomplete |
| `supersededCount` | ❌ MISSING | Safe (must be 0 in Phase I) but equation must include |
| `admittedLogicalObligationCount` | ❌ MISSING | MUST implement |
| `stalledCount` | ❌ MISSING | MUST implement (D10 stall ownership) |
| Ownership conservation equation | ❌ BROKEN (no successCount) | **BLOCKING** — D15 formula collapses |

---

## 4. Audit 3 — D18.4 Re-alignment (Reservation Semantics)

### Finding: `intentId → recoveryGeneration` conflation is the root bug

**D18.4**: `reservation` is acquired exactly once per newly created logical obligation, NOT per admission event. Coalesce does NOT acquire reservation.

### D103 Audit 3 Re-assessment

The D103 Audit 3 examined the reservation counter system:

```cpp
// ISRRuntimePublicationCoordinator.cpp:867 (CONFIRMED)
pendingRecoveryAdmission_.recoveryGeneration = intent.intentId;
// 🔴 D18.4 violation: intentId (diagnostic sequence) used as RecoveryGeneration (lineage serial)
```

### Current Reservation System Status

| Component | Status | Comment |
|---|---|---|
| `pendingIntentCount_` | ✅ Implemented | Transport reservation counter (fetchAdd/fetchSub with rollback) |
| `recoveryIntentDropCount_` | ✅ Implemented | Telemetry for queue full (INV-X1-3) — NOT ownership |
| `recoveryShutdownDiscardCount_` | ✅ Implemented | Terminal count for ShutdownDiscard |
| `PendingRecoveryAdmission.reservationOwned` | ✅ Implemented | Per-slot reservation flag |

### The `intentId → recoveryGeneration` Conflation (D8.3)

**Source code confirmation** (`ISRRuntimePublicationCoordinator.cpp:867`):
```cpp
pendingRecoveryAdmission_.recoveryGeneration = intent.intentId;  // 🔴 SEMANTIC CONFLATION
```

**Problem** (I2_DESIGN_FIX.md:22): `intentId` is a diagnostic sequence number (`nextRecoveryIntentId_`). Using it as `recoveryGeneration` (lineage serial) violates:
- D8.3: RecoveryGeneration must be from a **dedicated counter** (`nextRecoveryGeneration_`), not `intentId`
- D18.4: Reservation semantics require logical obligation identity to be independent of diagnostic sequence

**D16 correction**: RecoveryGeneration is `uint64_t`, 1..UINT64_MAX, NO wraparound in Phase I (unlike `rebuildRequestGeneration` which uses `std::atomic<int>` at `AudioEngine.h:2552`).

### Verdict

| Item | Status | Phase I Impact |
|---|---|---|
| `nextRecoveryGeneration_` counter | ❌ MISSING (0 hits) | **BLOCKING** — must separate from `intentId` |
| `RecoveryGeneration` domain separation | ❌ CONFUSED | MUST implement as independent domain |
| `pendingIntentCount_` reservation pattern | ✅ Implemented | OK (reservation-before-push) |
| `intentId → recoveryGeneration` assignment | ❌ CONFIRMED BUG | MUST fix (I2_DESIGN_FIX.md:22) |

---

## 5. Audit 4 — D18.5 Re-alignment (Domain Separation)

### Finding: 3 independent domains must be separated

**D13 (D18.5)**: Canonicalize `ConfigLineageId` → `RecoveryEpisodeId`. Three independent domains:

| Domain | Order | Used in supersession? |
|---|---|---|
| `RecoveryEpisodeId` | lineage identity | ✅ same episode = coalesce candidate |
| `RecoveryGeneration` | lineage ordering | ⚠️ isAfter check (newer) |
| `SemanticRecoveryTarget` | target equality | ✅ equality = coalesce target |

### D103 Audit 4 Re-assessment

The D103 Audit 4 searched for 13 identifiers. With D13/D18.5:

| Identifier | Production Hits | Status | D18.5 Fix |
|---|---|---|---|
| `LogicalRecoveryIdentity` | 0 | ❌ MISSING | MUST implement (handle + episode + gen + target) |
| `RecoveryEpisodeId` | 0 | ❌ MISSING | MUST implement (D13: rename from ConfigLineageId) |
| `RecoveryProvenance` | 0 | ❌ MISSING | MUST implement (Transport/Durable/Retry/Quarantine) |
| `RecoveryGeneration` | 0 | ❌ MISSING | MUST implement (separate counter, not intentId) |
| `canSupersede` | 0 | ❌ MISSING | **INACTIVE in Phase I** (extension point) |
| `isSemanticSuperset` | 0 | ❌ MISSING | Phase I: use `isSemanticTargetSuperset` (equality) |
| `isSemanticTargetSuperset` | 0 | ❌ MISSING | MUST implement (equality-based per D18.2) |
| `isDomainSuperset` | 0 | ❌ MISSING | MUST implement (necessary condition check) |
| `SupersessionDecision` | 0 | ❌ MISSING | Phase I: `SupersessionDecision` = {CanSupersede, CoalesceIdentity} |
| `SemanticRecoveryTarget` | 0 | ❌ MISSING | MUST implement (5 fields + equality) |
| `admittedLogicalObligationCount` | 0 | ❌ MISSING | MUST implement (D18.3) |
| `supersededCount` | 0 | ❌ MISSING | Must be 0 in Phase I (equation inclusion) |

### `buildInputHash` in SemanticRecoveryTarget Context (D18.5)

- `RuntimeBuildSnapshot` (h:37) has `buildInput` (BuildInput struct) but **no hash**
- `RuntimeBuildFingerprint` (h:38) has `irIdentityHash`, `convolutionConfigHash`, `dspParameterHash` but **no `buildInputHash`**
- D12.2 `SemanticRecoveryTarget` requires `buildInputHash` as 5th field
- **Fix**: Add `std::uint64_t buildInputHash` to `SemanticRecoveryTarget`, computed from `canonicalHash(buildInput)`

### `RecoveryGeneration` vs `rebuildRequestGeneration` (D8.3, D18.5)

- `recoveryGeneration`: allocated at admission time, identifies recovery lineage within episode
- `rebuildRequestGeneration`: `AudioEngine.h:2552` — `std::atomic<int>`, used as BOTH rebuild gen AND recovery gen
- D8.3: These must be **separated** — `buildSource.generation` is BuildGeneration (set at build completion), `recoveryGeneration` is lineage serial (set at admission)
- **CONFIRMED**: `ISRRuntimePublicationCoordinator.h:355` comment says `recoveryGeneration = intent.intentId` — semantic conflation

### Stalled Ownership (D10)

- **D10**: `Stalled` is a **live ownership category** (CoordinatorLoop owns bounded stall set)
- D103 found `stalledCount` = 0 hits
- `PendingRecoveryAdmission` has no `Stalled` state — only `NoAdmission/DurablePending/Building`
- D10: STALL must be bounded (`kMaxDurableRecoveryAdmissions` capacity), FIFO retry
- **Phase I must add**: `Stalled` state to `PendingRecoveryAdmission::State` enum, `recoveryStall_` bounded set

### Verdict

| Item | Status | Phase I Impact |
|---|---|---|
| RecoveryEpisodeId / ConfigLineageId | ❌ MISSING | MUST implement (D13) |
| RecoveryGeneration (dedicated counter) | ❌ MISSING | MUST implement (D8.2) |
| LogicalRecoveryIdentity struct | ❌ MISSING | MUST implement (R1) |
| RecoveryProvenance enum | ❌ MISSING | MUST implement |
| Stalled state in PendingRecoveryAdmission | ❌ MISSING | MUST implement (D10) |
| `intentId → recoveryGeneration` conflation | ❌ CONFIRMED | MUST fix |
| buildInputHash field | ❌ MISSING | **BLOCKING** (D12.2) |

---

## 6. Audit 5 — D18.6 Re-alignment (Test Matrix)

### Finding: Test matrix must be re-layered (unit → admission → integration)

**D18.6**: New test additions:
| # | テスト | 検証 |
|---|---|---|
| **T18** | identical target → **COALESCE** | 同 handle・同 episode・同 target → coalesce・reservation 不変・logicalObligation 不変 |
| **T22** | coalesce と budget 満杠 | budget 満杠でも既存同一 CoalesceIdentity への coalesce は成功（reservation 取得不要） |
| **T23** | reservation lifetime | SUPERSEDE: incoming +1 → older release（Superseded 後） / SUCCESS・SHUTDOWN_DISCARD: release |
| T15 | ownership conservation（修正） | liveOwnershipCount + terminalDispositionCount == admittedLogicalObligationCount（success 含む） |
| T15b | admissionEventCount ≥ admittedLogicalObligationCount | coalesce は event を増やすが logical 数を増やさない |

### D103 Audit 5 Re-assessment

Phase I tests T1-T17 must be re-ordered by **dependency**, not sequence number:

**Layer 1 — Unit (Identity/Semantic)**:
- T9: SemanticRecoveryTarget equality (5-field comparison)
- T11: RecoveryGeneration monotonic allocation
- T12: RecoveryEpisodeId monotonic allocation

**Layer 2 — Admission (Coalesce/BoundedTable)**:
- T18: identical target → COALESCE
- T22: coalesce + budget full (no reservation on coalesce)
- T24: backpressure progress / no deadlock

**Layer 3 — Integration (Ownership/Reservation)**:
- T15: ownership conservation (with successCount)
- T15b: admissionEventCount ≥ admittedLogicalObligationCount
- T23: reservation lifetime
- T25: episode closure finality
- T26: capacity bound (INV-CAP-1..4)
- T27: coalesce source mutation (buildSource update, identity immutable)

### Production Tests (Phase I GO prerequisite)

| Test | Location | Current State |
|---|---|---|
| `testRecoveryRequestEnqueueAndPop` | `ISRSemanticValidationTests.cpp:608` | ✅ Basic 1-hop transport verified |
| `testSubmitRecoveryRequestEpochPropagation` | `ISRSemanticValidationTests.cpp:665` | ✅ Epoch propagation verified |
| `testDurableRecoveryFallback` | `ISRSemanticValidationTests.cpp:688` | ✅ Durable fallback (single-slot overwrite) verified |

**Missing tests (Phase I GO prerequisite)**: T18 (coalesce), T22 (coalesce + budget), T15 (ownership conservation w/ successCount).

### Verdict

| Item | Status | Phase I Impact |
|---|---|---|
| T18 (coalesce) | ❌ MISSING | MUST implement (Layer 2) |
| T22 (coalesce + budget) | ❌ MISSING | MUST implement (Layer 2) |
| T15 (ownership conservation) | ❌ MISSING | MUST fix + implement (Layer 3) |
| T15b, T23, T24, T25, T26, T27 | ❌ MISSING | MUST implement (Layer 3) |
| T1-T17 re-ordered by dependency | ✅ Design frozen | Re-order implementation checklist |

---

## 7. Audit 6 — Production Change Target Files (Read-Only Inventory)

### Files That Must Be Modified for Phase I Implementation

| File | Changes Required |
|---|---|
| `src/audioengine/ISRRuntimePublicationCoordinator.h` | Add `LogicalRecoveryIdentity`, `RecoveryEpisodeId`, `Recovery_generation` counter, `SemanticRecoveryTarget`, `ObligationDomains`, `RecoveryProvenance`, `Stalled` state to `PendingRecoveryAdmission`, ownership counters (`successCount`, `supersededCount`, `stalledCount`, `admittedLogicalObligationCount`), bounded durable table (`kMaxDurableRecoveryAdmissions`) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | Fix `submitRecoveryRequest`: replace blind overwrite with `admitDurable()` (coalesce search → supersede → insert → stall); fix `intentId → recoveryGeneration` conflation; separate `nextRecoveryGeneration_` counter; add ownership counter increments/decrements |
| `src/audioengine/RuntimeBuildTypes.h` | Add `buildInputHash` to `RuntimeBuildFingerprint` or `SemanticRecoveryTarget`; add `canonicalHash(BuildInput)` function |
| `src/audioengine/AudioEngine.h` | Separate `rebuildRequestGeneration` (BuildGeneration) from RecoveryGeneration; add `nextRecoveryEpisodeId_` counter |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp` | Update recovery build path to emit `RecoveryGeneration` (not `generation`); separate from BuildGeneration (D8.3) |

### Files That Must NOT Be Changed (Forbidden Implementations)

Per D18 and user correction #15:
- **No `canSupersede()` implementation** in Phase I — it is a Phase II extension point
- **No partial-order semantic containment** — Phase I uses equality only (D18.2)
- **No `buildInputHash` bypass** — must be computed from `BuildInput` canonicalization

---

## 8. D18 Blocking Conditions (Phase I NO-GO Until Resolved)

### Determined Blocking Items

| # | Blocking Item | Description | Status |
|---|---|---|---|
| E | `kMaxLogicalRecoveryObligations` value | Must be derived from E_max × O_max ≤ 32 (not arbitrary). Requires upstream bound analysis of concurrent quarantine targets. | ❌ UNRESOLVED |
| F | E_max / O_max derivation | Must prove max concurrent quarantine targets × max per-episode distinct targets ≤ 32. | ❌ UNRESOLVED |
| G | Backpressure liveness proof | Must prove: park (CoordinatorLoop) → Builder (independent) releases reservation → pending becomes admissible. No circular dependency. | ❌ UNRESOLVED |
| H | Episode closure linearization proof | Must prove: atomic `liveLogicalObligationCount` + `Closed` flag provide total order for admission/commit/terminal disposition. | ❌ UNRESOLVED |

### D18.6 Test Matrix: Phase I GO Conditions

Phase I implementation may begin when ALL of:

1. **T18** (coalesce on identical target) implemented and passing
2. **T15** (ownership conservation with `successCount`) implemented and passing
3. **T22** (coalesce + budget full) implemented and passing
4. **D/E/F resolved**: `kMaxLogicalRecoveryObligations = 32`, E_max, O_max, backpressure liveness, closure linearization all proven

---

## 9. Final Verdict — Phase I Implementation Readiness

### Verdict: ⚠️ CONDITIONAL GO (3 blocking items remain)

| Category | Status | Detail |
|---|---|---|
| **D18.1** (Coalesce boundary) | ✅ FIXED | Phase I = coalesce-centered, supersede = Phase II |
| **D18.2** (Phase I semantics) | ✅ FIXED | Equality-based containment, not partial-order |
| **D18.3** (Ownership conservation) | ⚠️ REQUIRES IMPLEMENTATION | `successCount` missing from production counters |
| **D18.4** (Reservation semantics) | ⚠️ REQUIRES FIX | `intentId → recoveryGeneration` conflation (I2_DESIGN_FIX.md:22) |
| **D18.5** (D13-D17 revision) | ✅ FIXED | RecoveryEpisodeId canonicalized, wraparound excluded |
| **D18.6** (Test matrix) | ⚠️ REQUIRES IMPLEMENTATION | T18, T22, T15 missing; T1-T17 re-ordered by dependency |
| **E** (budget value) | ❌ BLOCKING | `kMaxLogicalRecoveryObligations = 32` needs derivation proof |
| **F** (capacity derivation) | ❌ BLOCKING | E_max × O_max ≤ 32 proof required |
| **G** (backpressure liveness) | ❌ BLOCKING | No deadlock proof required |
| **H** (closure linearization) | ❌ BLOCKING | Atomic counter + Closed flag total order proof |

### D103 vs D104 Finding Shift

| D103 Focus | D104 (D18-corrected) Focus |
|---|---|
| "canSupersede() 0 hits → NO-GO" | "Phase I doesn't need canSupersede → real blocker is coalesce infra" |
| "Supersession gate" | "Coalesce gate + 3-domain identity separation" |
| "Ownership equation broken (D15)" | "Ownership equation incomplete (missing successCount / stalledCount / admittedLogicalObligationCount)" |
| "buildInputHash missing" | "buildInputHash missing — Phase I semantic target field" |
| "Single-slot overwrite = NO-GO" | "Single-slot overwrite = NO-GO (same finding, D18 context)" |

### Implementation Gate (Stopping Condition)

> **D104 が PASS して初めて production source への実装を開始するのが安全です。**
>
> Phase I implementation GO gate:
> 1. D18.1–D18.6 contract fixed and documented (✅ this audit)
> 2. E/F blocking items resolved (kMax=32, E_max, O_max derived)
> 3. G (backpressure liveness) proof established
> 4. H (closure linearization) proof established
> 5. Phase I implementation checklist populated (D104_FINAL_IMPLEMENTATION_CONTRACT.md)
>
> **このGateがPASSするまで次の実装を禁止** — No production source changes until D104_FINAL_IMPLEMENTATION_CONTRACT.md is created and all 4 gate conditions are met.

---

## Appendix A — D103 Findings Re-assessed Against D18

### Audit 0 (Re-assessed): RecoveryIntent structure
- D103: `RecoveryIntent` has `DSPHandle handle`, `PublicationEpoch epoch`, `uint64_t intentId`, `RuntimeBuildSnapshot buildSource`
- D18: `RecoveryEpisodeId` MISSING from `RecoveryIntent` — MUST add. `RecoveryGeneration` must be separate counter, not `intentId`.
- **CONFIRMED**: `ISRRuntimePublicationCoordinator.h:216-228` — `RecoveryIntent` struct lacks `RecoveryEpisodeId` and `RecoveryGeneration`.

### Audit 1 (Re-assessed): PendingRecoveryAdmission single-slot
- D103: Single-slot struct, blind overwrite
- D18: MUST extend to bounded durable table (`kMaxDurableRecoveryAdmissions`), add coalesce search, separate `RecoveryGeneration`
- **CONFIRMED**: `ISRRuntimePublicationCoordinator.h:654-681` — single-slot, no coalesce.

### Audit 2 (Re-assessed): Semantic target containment
- D103: `isRuntimeBuildSnapshotSealedAndCompatible()` is equality, `buildInputHash` missing
- D18: Phase I uses equality-based `isSemanticTargetSuperset()` on `SemanticRecoveryTarget` (5 fields)
- **CONFIRMED**: `RuntimeBuildTypes.h:301-337` — equality check, no `buildInputHash`.

### Audit 3 (Re-assessed): Domain separation
- D103: 13 identifiers all 0 hits
- D18: `RecoveryEpisodeId` (D13), `RecoveryGeneration` (separate domain, D16), `SemanticRecoveryTarget` (D12.2)
- **CONFIRMED**: `recoveryGeneration = intent.intentId` at `cpp:867` — semantic conflation.

### Audit 4 (Re-assessed): Ownership counters
- D103: `pendingIntentCount_` ✅, `recoveryShutdownDiscardCount_` ✅, `successCount` (test-only), `supersededCount` (0), `admittedLogicalObligationCount` (0)
- D18: Must add `successCount`, `stalledCount`, `admittedLogicalObligationCount` to production; `stalledCount` needs D10 Stalled state
- **CONFIRMED**: Only `recoveryShutdownDiscardCount_` and `recoveryIntentDropCount_` exist in production.

### Audit 5 (Re-assessed): submitRecoveryRequest blind overwrite
- D103: Blind overwrite, no coalesce/supersession/identity check
- D18: Must implement `admitDurable()` with Step 0 (CoalesceIdentity search) → Step 1 (canSupersede, Phase II) → Step 2 (insert) → Step 3 (stall)
- **CONFIRMED**: `ISRRuntimePublicationCoordinator.cpp:865-872` — blind overwrite.

---

## Appendix B — R1-R17 Dependency-Ordered Implementation Checklist

### Layer 1 — Identity & Semantic Domains (Independent)

| R# | Requirement | Depends On | Production File |
|---|---|---|---|
| D16 | RecoveryGeneration: uint64, no wraparound, dedicated counter | — | ISRRuntimePublicationCoordinator.h |
| D13 | RecoveryEpisodeId: lineage identifier, monotonic | — | ISRRuntimePublicationCoordinator.h |
| D12.2 | SemanticRecoveryTarget: 5 fields + equality | D16 | RuntimeBuildTypes.h + ISRRuntimePublicationCoordinator.h |
| D10 | Stalled state: bounded stall set, FIFO retry | — | ISRRuntimePublicationCoordinator.h |
| R11 | transient failure restores durable obligation | — | ISRRuntimePublicationCoordinator.cpp |
| R12 | successful build consumes exactly one logical obligation | — | ISRRuntimePublicationCoordinator.cpp |

### Layer 2 — Admission & Coalesce (Depends on Layer 1)

| R# | Requirement | Depends On | Production File |
|---|---|---|---|
| R1 | LogicalRecoveryIdentity (handle + episode + gen + target) | D13, D16, D12.2 | ISRRuntimePublicationCoordinator.h |
| D4' | canSupersede() (Phase II extension point — NOT implemented in Phase I) | R1 | ISRRuntimePublicationCoordinator.h |
| R8-R10 | Supersession decision conditions | R1 | ISRRuntimePublicationCoordinator.h |
| R13 | queue-full does not lose recovery (durable fallback) | — | ISRRuntimePublicationCoordinator.cpp |
| R14 | coalesce does not increase reservation | R1 | ISRRuntimePublicationCoordinator.cpp |
| R15 | coalesce does not delete non-supersedable recovery | R1 | ISRRuntimePublicationCoordinator.cpp |

### Layer 3 — Ownership & Conservation (Depends on Layer 2)

| R# | Requirement | Depends On | Production File |
|---|---|---|---|
| D18.3 | Ownership conservation equation (live + terminal = admitted) | R12, R13 | ISRRuntimePublicationCoordinator.h |
| D18.4 | Reservation semantics (exactly once per new obligation) | R14, R15 | ISRRuntimePublicationCoordinator.cpp |
| R16 | shutdown closes RecoveryAdmission | D18.4 | ISRRuntimePublicationCoordinator.cpp |
| R17 | BuilderStopped participates in shutdown proof | R16 | ISRRuntimePublicationCoordinator.cpp |
| D14 | Capacity/reservation-first/backpressure | R1-R15 | ISRRuntimePublicationCoordinator.h |
| INV-X1-5 | 1 logical = exactly 1 reservation | D18.4 | ISRRuntimePublicationCoordinator.cpp |

### Layer 4 — Tests (Depends on Layer 3)

| T# | Test | Layer | Phase I GO Req |
|---|---|---|---|
| T1-T8 | Semantic domain tests | 1 | ✅ |
| T9 | SemanticRecoveryTarget equality | 1 | MUST |
| T10 | LogicalRecoveryIdentity generation | 1 | MUST |
| T11 | RecoveryGeneration monotonic | 1 | MUST |
| T12 | RecoveryEpisodeId monotonic | 1 | MUST |
| **T18** | Identical target → COALESCE | 2 | **MUST** |
| T13-T14 | Reservation admission | 2 | MUST |
| **T22** | Coalesce + budget full | 2 | **MUST** |
| T24 | Backpressure progress | 2 | MUST |
| **T15** | Ownership conservation (fixed) | 3 | **MUST** |
| T15b | admissionEventCount ≥ admittedLogicalObligationCount | 3 | MUST |
| **T23** | Reservation lifetime | 3 | MUST |
| T25 | Episode closure finality | 3 | MUST |
| T26 | Capacity bound | 3 | MUST |
| T27 | Coalesce source mutation | 3 | MUST |
| T16-T17 | Shutdown | 4 | MUST |

---

## Appendix C — Production Change Prohibited (Forbidden List)

Per D18 and user correction #15 (禁止実装):

| Forbidden | Reason |
|---|---|
| `canSupersede()` implementation in Phase I | D18.1: Phase I uses equality-based coalesce; supersede is Phase II extension point |
| Partial-order `isSemanticSuperset` (compositional containment) | D18.2: Phase I uses `semanticTargetEqual` (all 5 fields equal only) |
| `buildInputHash` bypass (using individual BuildInput field comparison) | D12.2: `SemanticRecoveryTarget.buildInputHash` is a required field |
| `intentId` reuse as `RecoveryGeneration` | D8.3: RecoveryGeneration must be from dedicated `nextRecoveryGeneration_` counter |
| `wraparound` in RecoveryGeneration | D16: NOT in Phase I (overflow is lifetime-bound unreachable) |
| `ConfigLineageId` as separate type | D13: Canonicalized to `RecoveryEpisodeId` only |
| Silent discard of non-coalescable recovery | D18.1 Step 3/4: STALL, never drop |

---

*D104 created by re-auditing D103 findings against D18.1–D18.6, D19–D22 of I4_DESIGN_CONTRACT.md.*

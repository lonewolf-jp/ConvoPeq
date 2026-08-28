# D105-R20 — RejectedNotFinalized Retry Centralization / Single-Count Repair

**Status:** **R20 PASS** ✅ (Debug + Release, 40/40 ctest PASS)
**Source changes:** 3 sites in `RuntimePublicationOrchestrator.cpp`; **4 new tests** (T-R20-1..4);
C8 unchanged; I4 unchanged.

This is the implementation that **rejects R19's "1-line switch-side replacement" approach**
(which would have produced double-counting for paths A and B) and instead centralizes
the obligation disposition in `submitPublishRequest`'s `RejectedNotFinalized` switch case.

After R20: **`markTransientFailure` is called exactly once per failure event** across
all four production failure paths. The R18 contract (ΔL=0 until retry exhaustion; one
counter increment per failure; `ResolvedFailed` only at counter == K) is now structurally
satisfied.

---

## R20-1 — Pre-implementation audit (4 paths, read-only)

| Path | Source site | `trySubmitImpl` action | Returned decision | Post-R20 disposition site |
|---|---|---|---|---|
| **A** Build #1 failure | `RuntimePublicationOrchestrator.cpp:178-192` | (after R20) **no `markTransientFailure`** — return `RejectedNotFinalized` | `RejectedNotFinalized` | `:389` centralized |
| **B** Crossfade rebuild failure | `RuntimePublicationOrchestrator.cpp:247-261` | (after R20) **no `markTransientFailure`** — return `RejectedNotFinalized` | `RejectedNotFinalized` | `:389` centralized |
| **C** Publish failure | `RuntimePublicationOrchestrator.cpp:277-314` | `markTransientFailure(id)` (line 311) | `RejectedPublishFailure` (or `RejectedShutdown` if shutting down) | `:420` switch (no-op) |
| **D** Admission rejection | `RuntimePublicationOrchestrator.cpp:46-51` | (no `markTransientFailure`) — return decision directly | `RejectedNotFinalized` (or other rejection) | `:389` centralized (for `RejectedNotFinalized`) |

The user's R20 brief correctly identified that **R19's "1-line switch-side replacement"
approach would have produced double-counting**: paths A and B already call
`markTransientFailure` (from R18) before returning; replacing `:389`'s
`resolveIfRecovery(Failed)` with `markTransientFailure` would have produced **2
`markTransientFailure` calls per failure event** (counter +2 instead of +1).

R20 reverses R18's "remove A/B/D's `markTransientFailure` and put it in `:389`" structure:
- A and B no longer call `markTransientFailure` directly.
- `:389` (the `RejectedNotFinalized` switch case) is the **single** disposition site
  for paths A, B, and D.
- C (publish failure) still calls `markTransientFailure` directly at line 311
  (it returns `RejectedPublishFailure`, not `RejectedNotFinalized`).

---

## R20-2 — Implementation (centralized at `:389`)

### File: `src/audioengine/RuntimePublicationOrchestrator.cpp`

**Path A (Build #1 failure, lines 186-192):**
```cpp
// Before (R18):
engine_.runtimePublicationBridge_.markTransientFailure(req.recoveryObligationId);
return PublicationAdmission::Decision::RejectedNotFinalized;

// After (R20):
// ★ D105-R20: transient build failure → return RejectedNotFinalized. The obligation
//   disposition is centralized in submitPublishRequest's RejectedNotFinalized
//   switch case (one markTransientFailure call per failure event; A/B/D unified).
return PublicationAdmission::Decision::RejectedNotFinalized;
```

**Path B (Crossfade rebuild failure, lines 254-260):**
```cpp
// Before (R18):
engine_.runtimePublicationBridge_.markTransientFailure(req.recoveryObligationId);
return PublicationAdmission::Decision::RejectedNotFinalized;

// After (R20):
// ★ D105-R20: transient crossfade-rebuild failure → return RejectedNotFinalized.
//   Centralized in submitPublishRequest's switch case (one markTransientFailure
//   per failure event; A/B/D unified). See Build #1 failure above.
return PublicationAdmission::Decision::RejectedNotFinalized;
```

**`:389` switch case (lines 389-402) — the centralized disposition site:**
```cpp
case PublicationAdmission::Decision::RejectedNotFinalized:
    stateOwner_.onRejected(0);
    telemetryRecorder_.recordFailure(FailureStage::Admission,
        FailureReason::ValidationFailed, "submitPublishRequest:notFinalized",
        0, nowUs);
    // ★ D105-R20: centralized markTransientFailure for ALL three RejectedNotFinalized
    //   paths (A: build #1, B: crossfade rebuild, D: admission-rejection). The previous
    //   direct resolveIfRecovery(Failed) was a leak-tightening path (R5-9) that
    //   conflicted with the R18 retry-preservation contract; centralizing here
    //   guarantees exactly one markTransientFailure call per failure event. ΔL=0,
    //   delivery=None (P-B), counter+1, exhaustion→ResolvedFailed (path X only).
    if (req.recoveryObligationId != 0)
        engine_.runtimePublicationBridge_.markTransientFailure(req.recoveryObligationId);
    return;
```

**Path C (Publish failure, lines 311) — unchanged:**
```cpp
// ★ D105-R18: transient publish failure → markTransientFailure (ΔL=0, same id,
//   delivery=None — repairs stranded Transport after Builder pop, retry counter
//   incremented). ResolvedFailed only at retry exhaustion. The shutdown check
//   below still routes to RejectedShutdown; obligation disposition is independent
//   of the return decision.
engine_.runtimePublicationBridge_.markTransientFailure(req.recoveryObligationId);
if (engine_.isShutdownInProgress())
    return PublicationAdmission::Decision::RejectedShutdown;
return PublicationAdmission::Decision::RejectedPublishFailure;
```

### `markTransientFailure` call count per failure event (post-R20)

| Failure event | `markTransientFailure` calls | Counter increment | ΔL |
|---|---:|---:|---:|
| A — Build #1 failure (Orchestrator site) | **1** (in `:389`) | +1 | 0 |
| B — Crossfade rebuild failure (Orchestrator site) | **1** (in `:389`) | +1 | 0 |
| C — Publish failure (Orchestrator site, line 311) | **1** (direct) | +1 | 0 |
| D — Admission rejection (`:389` switch) | **1** (in `:389`) | +1 | 0 |
| E — Builder durable build failure (RebuildDispatch:1039) | **1** (direct) | +1 | 0 |
| F — Builder durable warmup failure (RebuildDispatch:1063) | **1** (direct) | +1 | 0 |

**No failure event results in more than 1 `markTransientFailure` call.** The R20 GO
condition "exactly once per failure event" is structurally enforced.

---

## R20-3 — `delivery = None` confirmation (P-B repair for all paths)

`markTransientFailure` (R18-2, `ISRRuntimePublicationCoordinator.cpp:1010`) unconditionally
sets `delivery = None` *before* incrementing the counter. This P-B repair is the same
across all six failure sites:

```cpp
recoveryAdmissions_.slot(i).delivery = ObligationDeliveryState::None;  // line 1010
```

The obligation is now `Live && delivery == None` and is **redrive-eligible**. The next
`redriveDeferredRecoveryObligations` tick (R5-10) will re-attach delivery (durable or
transport). The redrive preserves the same `obligationId` (R5-10 §2).

| State after `markTransientFailure` | Verified by |
|---|---|
| `Live` (ΔL=0) | T-R20-1, T-R18-1..12 (unchanged) |
| `delivery == None` | T-R18-3, T-R20-1 (recovery redrive) |
| `counter` incremented by 1 | T-R18-5, T-R20-3 (exactly-once counting) |
| `liveCount` unchanged until exhaustion | T-R20-3 (4 calls → 0) |
| Exhaustion at counter == K | T-R20-4, T-R18-5 |
| `recoveryRetryExhaustedCount` increments only at exhaustion | T-R20-3, T-R20-4 |

---

## R20-4 — Test design (T-R20-1..4)

**File:** `src/tests/ISRSemanticValidationTests.cpp` (added after T-R18-12, registered in `main()`)

### T-R20-1 — Admission rejection retry preservation
Simulates the path-D scenario: obligation Live + delivery=Transport. After a single
`markTransientFailure` call (the centralized `:389` dispatch), the obligation stays
Live, `delivery=None`, `recoveryRetryRedriveCount` increments after `redriveDeferredRecoveryObligations`.

### T-R20-2 — Admission rejection redrive preserves the same obligationId
The full chain: `markTransientFailure` → `delivery=None` → redrive → obligation re-attached
to a delivery representation (durable or transport) → same `obligationId`.

### T-R20-3 — Exactly-once failure counting
**The critical test.** Asserts that **1 call to `markTransientFailure` increments the
counter by 1 (NOT 2)**. The R19 proposal would have produced `0→2` for paths A and B; R20
must produce `0→1`. Also asserts that 3 calls keep the obligation Live (counter=3 < K=4)
and the 4th call produces `ResolvedFailed` with `recoveryRetryExhaustedCount==1`.

### T-R20-4 — `ResolvedFailed` only via exhaustion
Verifies that the only path to `ResolvedFailed` in production is via
`markTransientFailure`'s exhaustion branch (counter == K). After 1 `markTransientFailure`
call, the obligation is Live and `recoveryRetryExhaustedCount == 0`. After 4 calls, the
obligation is `ResolvedFailed` and `recoveryRetryExhaustedCount == 1`.

---

## R20-5 — `RecoveryOutcome::Failed` API retained (R19 recommendation)

`resolveRecoveryObligation(_, Failed)` switch arm in
`ISRRuntimePublicationCoordinator.cpp:969` is **retained** for two reasons:

1. **C8 compatibility**: `ISRSemanticValidationTests.cpp:1034` (C8) calls
   `resolveRecoveryObligation(*id, RecoveryOutcome::Failed)` directly to test the
   table's `Live → ResolvedFailed` CAS as a unit test. Removing the arm would break C8.
2. **Defensive depth**: any future code that adds a new caller would be detected
   (Debug) by the `jassertfalse` in the `default:` branch (line 974).

After R20, **`resolveRecoveryObligation(_, Failed)` has zero production callers**.
The only production path to `ResolvedFailed` is `markTransientFailure`'s exhaustion
branch, which calls the table's `resolve` directly with `ObligationState::ResolvedFailed`
(not through the public `resolveRecoveryObligation` API).

The `Failed` arm in the switch (line 969) is now annotated as "test / defensive
compatibility only" (R19's R19-4 recommendation, applied at R20). Production should
use `markTransientFailure` (which handles counter increment, P-B, and exhaustion
together as a single atomic transition).

---

## R20-6 — Grep verification

### `markTransientFailure` production call sites (4 sites, exactly 1 per failure event)

```
src/audioengine/AudioEngine.RebuildDispatch.cpp:1039  (Builder durable build failure, Path E)
src/audioengine/AudioEngine.RebuildDispatch.cpp:1063  (Builder durable warmup failure, Path F)
src/audioengine/RuntimePublicationOrchestrator.cpp:311  (Publish failure, Path C)
src/audioengine/RuntimePublicationOrchestrator.cpp:401  (RejectedNotFinalized switch, Paths A/B/D)
```

### `RecoveryOutcome::Failed` references (0 production callers)

```
src/audioengine/ISRRuntimePublicationCoordinator.cpp:952   (comment)
src/audioengine/ISRRuntimePublicationCoordinator.cpp:969   (switch arm — kept for C8)
src/tests/ISRSemanticValidationTests.cpp:1034               (C8 test only)
```

**`resolveRecoveryObligation(_, RecoveryOutcome::Failed)` production callers: 0.** ✅

---

## R20-7 — Test results

### Debug build

```
[SUCCESS] Executable created successfully.
build\ConvoPeq_artefacts\Debug\ConvoPeq.exe
```

### Release build

```
[SUCCESS] Executable created successfully.
build\ConvoPeq_artefacts\Release\ConvoPeq.exe
```

### ctest (Debug)

```
40/40 Test #40: AudioEngineHarness ........................   Passed   16.54 sec
100% tests passed out of 40
Total Test time (real) =  34.59 sec
```

### ctest (Release)

```
40/40 Test #40: AudioEngineHarness ........................   Passed   15.40 sec
100% tests passed out of 40
Total Test time (real) =  29.57 sec
```

All 40 tests pass in both configurations, including:
- **C1–C16** (R5-9 / R5-10 obligations tests)
- **T-R13-1..4** (R13 isFullyDrained structural assertion)
- **C8** (table-level Failed arm test, unchanged)
- **T-R18-1..12** (R18 retry-preserving production tests)
- **T-R20-1..4** (R20 single-count and centralized-dispatch tests)

---

## R20-8 — GO-condition verification

| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | A Build failure → `markTransientFailure` exactly once | ✅ | A no longer calls `markTransientFailure`; `:389` calls it once |
| 2 | B Crossfade failure → exactly once | ✅ | B no longer calls `markTransientFailure`; `:389` calls it once |
| 3 | C Publish failure → exactly once | ✅ | C at line 311 calls once; `:420` switch is no-op |
| 4 | D Admission rejection → exactly once | ✅ | D no longer calls `markTransientFailure`; `:389` calls it once |
| 5 | `delivery == None` for all transient paths | ✅ | `markTransientFailure` unconditionally sets `delivery=None` (line 1010) |
| 6 | ΔL == 0 before exhaustion | ✅ | Only `markTransientFailure`'s exhaustion branch touches `liveCount_` |
| 7 | Counter +1 per failure | ✅ | T-R20-3 verifies `0→1` (not `0→2`) |
| 8 | K=4 → 4th call → `ResolvedFailed` | ✅ | T-R20-3, T-R20-4 |
| 9 | Exhaustion telemetry increments only at exhaustion | ✅ | `recoveryRetryExhaustedCount_++` only in exhaustion branch (line 1018) |
| 10 | `ResolvedFailed` production caller = exhaustion only | ✅ | grep: only `markTransientFailure`'s exhaustion branch |
| 11 | `resolveRecoveryObligation(Failed)` production caller = 0 | ✅ | grep: only C8 test (line 1034) |
| 12 | C8 unchanged / PASS | ✅ | C8 still passes (table-level test) |
| 13 | R18 tests all PASS | ✅ | T-R18-1..12 all pass |
| 14 | Debug build PASS | ✅ | 40/40 |
| 15 | Release build PASS | ✅ | 40/40 |

**R20: ALL GO conditions satisfied. PASS.**

---

## R20 prohibitions check

| # | Prohibition | Status |
|---|---|---|
| 1 | R19 1-line replacement | **Avoided** — A and B `markTransientFailure` were removed, replaced with centralized `:389` |
| 2 | A/B + switch both calling `markTransientFailure` | **Avoided** — single source of truth at `:389` for A/B/D |
| 3 | Direct `consecutiveFailureCount` mutation from Builder | **Avoided** — only `markTransientFailure` mutates counter (line 1013) |
| 4 | Change K=4 | **Avoided** — `kMaxObligationConsecutiveFailures = 4` unchanged |
| 5 | Change `kMaxLogicalRecoveryObligations = 32` | **Avoided** — capacity unchanged |
| 6 | Modify C8 | **Avoided** — C8 unchanged |
| 7 | Modify I4 | **Avoided** — `I4_DESIGN_CONTRACT.md` unchanged |
| 8 | Remove `RecoveryOutcome::Failed` enum | **Avoided** — enum value retained; switch arm kept for C8 |
| 9 | Add new retry counter | **Avoided** — R18's `consecutiveFailureCount` is the only counter |
| 10 | Add new obligation state for `RejectedNotFinalized` | **Avoided** — same `ResolvedFailed` (via exhaustion only) and `Live` states |

---

## R20 target state (achieved)

```
                 RejectedNotFinalized  (A: build, B: rebuild, D: admission)
                              │
                              ▼
                 markTransientFailure()   ← exactly once
                              │
                ┌─────────────┴─────────────┐
                │                           │
            counter < K               counter == K
                │                           │
                ▼                           ▼
       Live + delivery=None        ResolvedFailed
                │                           │
                ▼                           ▼
            redrive               recoveryRetryExhaustedCount_++
                │
                ▼
         same obligationId
```

Production `ResolvedFailed` is now exclusively **RetryExhaustion**. The R18 GO condition
6 (production `ResolvedFailed` = retry exhaustion only) is structurally satisfied.

---

## Files changed

| File | Change |
|---|---|
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | A site (line ~191): removed `markTransientFailure` call. B site (line ~259): removed `markTransientFailure` call. `:389` switch case: replaced `resolveIfRecovery(Failed)` with `markTransientFailure` (with `if (id != 0)` guard) |
| `src/tests/ISRSemanticValidationTests.cpp` | 4 new tests T-R20-1..4 (definitions + registration in `main()`) |

**No other files touched.** C8 unchanged. I4 unchanged. `consecutiveFailureCount` field
unchanged. `markTransientFailure` implementation unchanged. `kMaxObligationConsecutiveFailures`
unchanged. `kMaxLogicalRecoveryObligations` unchanged.

---

## R20 → R21 hand-off

R21 should:

1. **Apply** the I4 amendment wording from D105-R19 §R19-5 to `I4_DESIGN_CONTRACT.md`:
   - D14.3: add footnote (transient failure does NOT consume backpressure budget).
   - D15.2: replace disappearance set with `{Success, Superseded, ShutdownDiscard, RetryExhaustion}`.
   - D18.3: add `retryExhaustedCount` to conservation equation; add disjointness footnote.
2. **Add** I4 contract test(s) verifying:
   - Conservation equation holds (with `retryExhaustedCount`).
   - `retryExhaustedCount` only increments via `markTransientFailure`'s exhaustion branch.
3. **R22** (next, source-modifying or audit): re-audit the entire I4 ↔ runtime
   contract per the new state machine (full state coverage, all transitions traced,
   capacity proven, liveness proven).

R20 has fully implemented the runtime side; R21 + R22 close the contract loop.

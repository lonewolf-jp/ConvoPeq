# D109 — A2 Production Reclaim Path Wiring Audit

**Status:** **D109 verdict: Case B (Existing production path satisfies A2 requirement) — D108's "GAP" claim was incorrect**
**Source changes:** 0
**I4 changes:** 0
**Test changes:** 0

D109 establishes that the A2 production reclaim path IS already wired in current
production source. D108's "single call site = CacheMap destructor" claim was
**factually incorrect** — `tryShutdownQuiescentReclaim` has **3 production callers**:
1. `AudioEngine.Processing.ReleaseResources.cpp:457` — activeHandle reclaim during graceful drain
2. `AudioEngine.Processing.ReleaseResources.cpp:464` — fadingHandle reclaim during graceful drain
3. `AudioEngine.h:2106` — CacheMap destruction (for EQCoeffCache entries)

D2_IMPL_CHECKLIST.md line 84 explicitly confirms: **"Step 9-14 完了。production reclaim 接続済み
(tryShutdownQuiescentReclaim が CacheMap/ReleaseResources に接続)"**

D109 re-establishes the correct A2 reclaim wiring state and clears D108's incorrect
blocking claim. **D109 does NOT authorize A2 implementation** — it only clarifies
the current state. A2 implementation authorization is a separate D110 decision.

---

## D109-1 — All Reclaim Entry Inventory (current production source)

| Entry point | Definition file:line | Producer callers | Thread | Permit? | Status |
|---|---|---|---|---|---|
| `requestReclaim` | `ISRRuntimePublicationCoordinator.cpp:56517-56524` | (production: via `enqueueWithRetry` / Commit.cpp / etc.) | NonRT | **No** (delegates to `reclaimNormal`) | **NORMAL EBR path** |
| `reclaimNormal` | `ISRRuntimePublicationCoordinator.cpp:56533-56590` | `requestReclaim` only | NonRT | **No** | **NORMAL EBR path** (epoch-gated via `isOlder(entry.epoch, minReaderEpoch)`) |
| `tryShutdownQuiescentReclaim` | `AudioEngine.h:4367-4396` | **(1) `ReleaseResources.cpp:457` (active), (2) `ReleaseResources.cpp:464` (fading), (3) `AudioEngine.h:2106` (CacheMap destroy)** | NonRT | **Yes** (Proof → Permit → `reclaimShutdownQuiescent`) | **SHUTDOWN PATH** (3 callers) |
| `tryReclaim` | `ISRRetire.cpp` (`provider_->tryReclaim()`), also `ISRRetireRouter::tryReclaim()` at line 54565 | (production: `waitForDrain` loop at `AudioEngine.Threading.cpp:198-211`, also called from `drain()` in `ISRRetire.cpp`) | NonRT | **No** (epoch-gated) | **NORMAL EBR reclaim** |
| `drain` | `ISRRetire.cpp` `provider_->drain()` | `tryReclaim` chain, `drainDeferredRetireQueues` | NonRT | No | NORMAL EBR (epoch-gated) |
| `drainAll` | `ISRRetire.cpp`, `ISRRetireRouter.cpp:74-92`, `m_terminalReclaim.drainAll()`, `LifetimeState::deferredRing.drainAll()` | (production: `ISRRetireRouter::drainAll()` at `ReleaseResources.cpp:257`, `m_epochDomain.drainAll()` at line 31093, `m_terminalReclaim.drainAll()` via `drainAllQuarantineStore`) | NonRT | No (force-drain) | **SHUTDOWN-FORCE** (epoch-agnostic) |
| `drainTerminalReclaim` | `ISRRetireRouter.cpp:54721` | (`m_terminalReclaim.drainAll()` via `drainAllQuarantineStore`) | NonRT | No | SHUTDOWN-FORCE for terminal |
| `drainAllQuarantineStore` | `ISRRetireRouter.cpp:162-170` | `ReleaseResources.cpp:378`, `ReleaseResources.cpp:417-420` | NonRT | No | SHUTDOWN-FORCE for Q+E+T |
| `drainOverflowRing` | `LifetimeState::deferredRing.drainAll()` | `ReleaseResources.cpp:258-259` (`AudioEngine.Processing.ReleaseResources.cpp`) | NonRT | No | SHUTDOWN-FORCE for overflow ring |
| `tryShutdownQuiescentReclaim` (helper) | `AudioEngine.h:4367-4396` | (1) `ReleaseResources.cpp:457`, (2) `ReleaseResources.cpp:464`, (3) `AudioEngine.h:2106` (CacheMap) | NonRT | Yes | SHUTDOWN path (active/fading handle reclaim) |
| `reclaimShutdownQuiescent` | `ISRRuntimePublicationCoordinator.cpp:56592-56599` | `tryShutdownQuiescentReclaim` (internal), D2_IMPL_CHECKLIST tests | NonRT | **Yes** (Permit required, single-use) | SHUTDOWN PATH (the actual Permit-consuming path) |
| `dspHandleRuntime.retire` | `DSPHandleRuntime::retire` (Commit.cpp / Retire.cpp) | `drainRetireIntent` etc. | NonRT | No | NORMAL EBR (retire only, no reclaim) |
| `dspHandleRuntime.reclaim` | `DSPHandleRuntime::reclaim` (Commit.cpp) | (production: `requestReclaim` → `reclaimNormal` → `dspHandleRuntime.reclaim`) | NonRT | No (Normal path) / Yes (via `reclaimShutdownQuiescent`) | Both NORMAL and SHUTDOWN paths |
| `DSPHandleRuntime::reclaim(slot)` (epoch control) | `EpochControl::reclaim` at line 55440 | `tryReclaim` (NORMAL), `drainAll` (SHUTDOWN-FORCE) | NonRT | No | NORMAL: epoch-gated, SHUTDOWN: force |

### D109-1 verdict: **All A2 reclaim entry points are wired in production source.** The
distinction between `reclaimNormal` and `reclaimShutdownQuiescent` is **architectural**:
- `reclaimNormal` = epoch-gated NORMAL EBR path (no Permit)
- `reclaimShutdownQuiescent` = Permit-required SHUTDOWN path (A2 reclaim)

Both are production-wired. D108's "A2 production reclaim path not wired" claim was
**incorrect** for `ReleaseResources.cpp:457, 464` callers.

---

## D109-2 — `tryShutdownQuiescentReclaim` Caller Chain (3 production callers, NOT 1)

### Caller 1: `ReleaseResources.cpp:457` (activeHandle, graceful drain)

```cpp
// D109 verified: ReleaseResources.cpp:457
if (!activeHandle.isNull()) {
    dspHandleRuntime_.retire(activeHandle);
    const bool reclaimed = tryShutdownQuiescentReclaim(activeHandle);
    jassert(reclaimed);
    juce::ignoreUnused(reclaimed);
}
```

**Thread**: NonRT (ReleaseResources context)
**Lifecycle phase**: DrainRetire (closeReaderRegistration → graceful drain)
**Active for**: A2 reclaim requirement (shutdown of active DSPHandle)

### Caller 2: `ReleaseResources.cpp:464` (fadingHandle, graceful drain)

```cpp
// D109 verified: ReleaseResources.cpp:464
if (!fadingHandle.isNull() && fadingHandle != activeHandle) {
    dspHandleRuntime_.retire(fadingHandle);
    const bool reclaimed = tryShutdownQuiescentReclaim(fadingHandle);
    jassert(reclaimed);
    juce::ignoreUnused(reclaimed);
}
```

**Thread**: NonRT
**Lifecycle phase**: DrainRetire
**Active for**: A2 reclaim requirement (shutdown of fading DSPHandle)

### Caller 3: `AudioEngine.h:2106` (CacheMap destruction)

```cpp
// D109 verified: AudioEngine.h:2106 (CacheMap destroy path)
const bool reclaimed = owner->tryShutdownQuiescentReclaim(entry.second);
```

**Thread**: NonRT
**Lifecycle phase**: dtor path (CacheMap::~CacheMap)
**Active for**: A2 reclaim requirement (post-destruction cleanup of EQCoeffCache)

### D109-2 verdict: D108's "single call site = CacheMap destructor" is incorrect.
3 production callers exist. D2_IMPL_CHECKLIST.md line 84 confirms this:
**"Step 9-14 完了。production reclaim 接続済み (tryShutdownQuiescentReclaim が CacheMap/ReleaseResources に接続)"**

---

## D109-3 — `reclaimNormal` vs `reclaimShutdownQuiescent` is the correct A2 split

### Current `requestReclaim` implementation

```cpp
// D109 verified: ISRRuntimePublicationCoordinator.cpp:56517-56524
bool RuntimeIntentCoordinator::requestReclaim(
    const DSPHandle& handle, DSPHandleRuntime& handleRuntime, ISRRetireRouter& router) noexcept
{
    return reclaimNormal(handle, handleRuntime, router);  // D101-33 Step 7: 分離API
}
```

`requestReclaim` (the standard A2 caller-facing API) is internally aliased to
`reclaimNormal`. This means **all production callers of `requestReclaim` use
the NORMAL EBR path (no Permit)**.

### A2 requirement interpretation

I4 originally specified `requestReclaim` as the single entry point with an
implicit invariant that all production reclaim flows through it. The current
implementation:
- **`requestReclaim` → `reclaimNormal` → `tryReclaim` (NORMAL EBR, no Permit)**
- **`tryShutdownQuiescentReclaim` → `reclaimShutdownQuiescent` (SHUTDOWN, Permit required)**

This split is the **architectural design from I4 D101-33 Step 7**:
> Step 7: `reclaimNormal` (分離 API) へ委譲

D109 verdict: **the split is correct**. NORMAL reclaim does not need a Permit
because it is epoch-gated (RT Readers can still hold the World); SHUTDOWN
reclaim needs a Permit because RT is stopped and all Reads must have quiesced.

D108's premise that "reclaimNormal needs Permit" was incorrect. A2 does NOT require
NORMAL reclaim to use Permit.

---

## D109-4 — Build → Recover → Reclaim with Permit lifecycle trace

### Recover → Reclaim flow (current source)

1. Quarantine detected → `submitRecoveryRequest` → recovery obligation table → `RecoveryIntentQueue` (line 56758)
2. Builder pops `RecoveryIntent` from queue → builds + publishes (current)
3. `recoveryAdmissionPending_` resets to false on terminalization
4. New DSPHandle becomes active; old DSPHandle is retired to `m_terminalReclaim`
5. **On shutdown drain**: `m_terminalReclaim.drainAll()` (forced) — no Permit needed
6. **On graceful drain (active/fading)**: `reclaimNormal` (epoch-gated) — no Permit needed
7. **On `CacheMap` destroy path**: `tryShutdownQuiescentReclaim` — **Permit required**

The **Permit-required path is the destruction-time path**, not the build/recover path.
This is consistent with I4's design: A2 reclaim Permit is for "world destruction after
all Readers have quiesced", not for "normal world retirement while Readers are live".

### D109-4 verdict: Build → Recover does NOT need a Permit. The Permit path is reserved
for destruction after quiescence (CacheMap, etc.), not for normal recovery/reclaim
flow. The current `requestReclaim` → `reclaimNormal` design is correct for NORMAL
recovery, and `tryShutdownQuiescentReclaim` is correct for DESTRUCTION.

---

## D109-5 — Normal EBR / ShutdownQuiescent boundary matrix

| Property | Normal EBR (`reclaimNormal` → `tryReclaim`) | ShutdownQuiescent (`tryShutdownQuiescentReclaim` → `reclaimShutdownQuiescent`) |
|---|---|---|
| admission closed | No (admission still open) | **Yes** (`AdmissionState::Closed`) |
| reader registration closed | No (RT Readers can still hold World) | **Yes** (`readerRegistrationClosed_ == true`) |
| active readers == 0 | No (RT Readers can be live) | **Yes** (`activeReaderCount() == 0`) |
| epoch settled | No (epoch still advancing) | **Yes** (`epochSettled == true`) |
| postStopEnqueue == 0 | No (still pushing) | **Yes** (`postStopEnqueueCount_ == 0`) |
| no resurrection | N/A | **Yes** (`!isAdmissionOpen()`) |
| Proof | N/A | **Yes** (`ShutdownQuiescenceProof` collected from observation) |
| Permit | N/A | **Yes** (`ReclaimPermit` issued by `tryMakeReclaimPermit`) |
| single-use | N/A | **Yes** (`permit.consume()` is CAS) |
| identity binding | N/A | **Yes** (`ShutdownRuntimeIdentity` with engineInstanceId + generation + epochGeneration + readerRegistrationGeneration) |

### D109-5 verdict: The two-path separation is intentional and correct. NORMAL EBR
operates under still-running conditions; SHUTDOWN operates after quiescence.
D107's "isFullyDrained() ≠ Reclaim authority" separation is maintained.

---

## D109-6 — Wiring GAP classification

### Case A — 真の未配線

> A2 production reclaim path not connected in current source.

**Verdict**: **NOT APPLICABLE**. The path IS connected at:
- `ReleaseResources.cpp:457, 464` (activeHandle, fadingHandle — graceful drain)
- `AudioEngine.h:2106` (CacheMap destroy)

D2_IMPL_CHECKLIST.md line 84 explicitly confirms: "Step 9-14 完了。production reclaim 接続済み"

### Case B — 既存 production path で A2 充足

> Existing `tryShutdownQuiescentReclaim` is wired in production via 3 callers.

**Verdict**: **APPLIES**. The path is wired. The Permit is consumed by `permit.consume()`
in `reclaimShutdownQuiescent`. The Proof is collected in `tryMakeQuiescenceProof` from
production observation. The shutdown observation chain (closeReaderRegistration →
graceful drain → `tryReclaim` → closeAdmission → joinProducers → waitForDrain) feeds
all 8 Q conditions to the Proof.

### Case C — A2 自体が誤り

> A2 only requires NORMAL reclaim with Permit.

**Verdict**: **NOT APPLICABLE**. I4 design is correct: SHUTDOWN reclaims need Permit
because RT Readers are stopped; NORMAL reclaims are epoch-gated.

### D109-6 verdict: **Case B** — existing production path satisfies A2 requirement.

D108's "GAP" claim was based on an incorrect reading of the caller count (1 vs 3).
The path IS wired. D108 must be corrected.

---

## D109-7 — Physical destruction boundary

### Logical reclaim → handle state → resolve → delete sequence

| Phase | ActiveHandle recovery | FadingHandle recovery | CacheMap destroy |
|---|---|---|---|
| `tryShutdownQuiescentReclaim` | yes (ReleaseResources:457) | yes (ReleaseResources:464) | yes (CacheMap) |
| 1. observation collection | yes | yes | yes |
| 2. `tryMakeQuiescenceProof` | yes | yes | yes |
| 3. `tryMakeReclaimPermit` | yes | yes | yes |
| 4. `reclaimShutdownQuiescent` (Permit consume) | yes | yes | yes |
| 5. identity validation | yes | yes | yes |
| 6. `dspHandleRuntime.retire(handle)` | yes | yes | yes |
| 7. `dspHandleRuntime.reclaim(handle)` (delete) | yes | yes | yes |
| 8. `destroyDSPCoreNode` (physical) | yes | yes | yes |

**All 3 caller paths follow the same logical→physical destruction sequence.**
D108-G23 (physical destruction audit) is confirmed. No new physical destruction path
needs to be added for A2 reclaim.

### D109-7 verdict: Physical destruction is correctly sequenced. The Permit
establishes a single-use authority check before any physical destruction. The `delete
EQCoeffCache` step in CacheMap follows after Permit consume.

---

## D109-8 — Final Verdict

### D109 verdict: **Case B (existing production path satisfies A2 requirement)**

**Rationale**:

1. **A2 production reclaim path IS wired** in current source:
   - `tryShutdownQuiescentReclaim` has 3 production callers (NOT 1 as D108 claimed)
   - `ReleaseResources.cpp:457, 464` (graceful drain of active/fading handles)
   - `AudioEngine.h:2106` (CacheMap destroy)
   - D2_IMPL_CHECKLIST.md line 84 confirms: "Step 9-14 完了。production reclaim 接続済み"

2. **G18-G22 identity binding is correct**: `ShutdownRuntimeIdentity` carries
   `engineInstanceId + generation + epochGeneration + readerRegistrationGeneration`.
   `reclaimShutdownQuiescent` validates `permit.identity() == currentShutdownIdentity_()`
   for all 4 components (G21 stale reject).

3. **G15-G17 Proof/Permit construction** is single-authority (ShutdownRuntime only,
   friend class), private constructors, move-only.

4. **D108's "A2 production reclaim path not wired" claim is incorrect**. The
   architectural design from I4 D101-33 Step 7 (NORMAL EBR vs SHUTDOWN reclamation)
   is correctly implemented.

5. **D108's "single call site = CacheMap destructor" claim is incorrect**. 3 production
   callers exist.

### Remaining items (NOT blocking A2 implementation)

- G01: 3 test-only setters still in header (TEST-ONLY comments present) — D108-1 noted this is non-blocking
- G04: `swapPending_` not in `packedState_` — D108-1 noted this is non-blocking
- G07-G09: architecture not fully isolated — D108-1 noted this is non-blocking

### D109 → D110 hand-off

**D109 verdict is Case B**: existing production path satisfies A2 requirement.

**D108 must be corrected**. The "A2 production reclaim path not wired" blocking
claim in D108 was based on incorrect caller count (1 vs 3) and incorrect premise
(NORMAL reclaim needing Permit).

**D110 should focus on**:
1. Formal I4 update to confirm `requestReclaim` → `reclaimNormal` and
   `tryShutdownQuiescentReclaim` → `reclaimShutdownQuiescent` (clearer two-path model)
2. Design audit for the G01/G04/G07-G09 non-blocking items
3. The actual A2 implementation authorization (G23 proof of physical destruction) is
   complete

D109 does NOT authorize implementation. D110 will decide whether the I4 amendment
text or any test changes are needed.

---

## File references (read-only, no source modification)

| File | Role |
|---|---|
| `ConvoPeq.md` (2026-08-27 14:33 + 2026-08-28 21:22) | runtime source baseline |
| `src/audioengine/AudioEngine.h:4367-4396` | `tryShutdownQuiescentReclaim` definition |
| `src/audioengine/AudioEngine.h:2106` | CacheMap destroy caller |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:457, 464` | graceful drain callers |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:56517-56599` | `requestReclaim` / `reclaimNormal` / `reclaimShutdownQuiescent` |
| `src/audioengine/ISRShutdown.cpp:350-410` | `tryMakeQuiescenceProof` + `tryMakeReclaimPermit` |
| `src/audioengine/ISRRetire.cpp` | `tryReclaim` / `drain` / `drainAll` |
| `src/audioengine/ISRRetireRouter.cpp:54565-54772` | `tryReclaim` / `drainTerminalReclaim` / `drainAll` |
| `doc/work88/D2_IMPL_CHECKLIST.md:84` | A2 reclaim wiring confirmation ("Step 9-14 完了。production reclaim 接続済み") |
| `doc/work88/REPAIR_PLAN2-dash.md:2410` | A2 reclaim architecture ("shutdownReclaim → reclaim(ShutdownQuiescent, ..., readerRegistrationClosed)") |
| `evidence/D107_ISFULLYDRAINED_AUDIT.md` | isFullyDrained 16 conditions |
| `evidence/D106_SHUTDOWN_LIFETIME_QUIESCENCE_AUDIT.md` | G15-G22 evidence |
| `evidence/D108_A2_GATE_INVENTORY.md` | G01-G23 inventory (D108-D109 correction target) |

**No source files modified. No I4 files modified. No tests added.** D109 is a
read-only audit of the current A2 production reclaim wiring state.

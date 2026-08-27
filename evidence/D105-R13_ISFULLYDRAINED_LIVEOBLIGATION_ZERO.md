# D105-R13 — `isFullyDrained()` Logical Obligation Zero (Structural Assertion)

**Status:** PASS. **Source change:** 1 structural-assertion line in `isFullyDrained()` + 4 regression tests.
**Prerequisite:** D105-R12 (CONDITIONAL PASS — `liveCount_==0` proven *logically*, not yet asserted structurally).

Goal: elevate the already-proven `liveCount_ == 0` guarantee from an *emergent* property of shutdown
ordering into a *direct* predicate check, in the sense requested by D105-R11 hardening recommendation #1.

---

## R13-1 — Minimal change (exactly one condition added)

`src/audioengine/ISRRuntimePublicationCoordinator.cpp`, inside
`ShutdownScheduler::isFullyDrained()` (cpp:506), immediately after the recovery durable check
`!coordinator_.recoveryAdmissionPending_` (cpp:549):

```cpp
&& !convo::consumeAtomic(coordinator_.recoveryAdmissionPersistent_, std::memory_order_acquire)
// ★ D105-R13 (INV-1-2 structural): logical obligation residency must be zero. Previously this held
//   only as an emergent property of shutdown ordering (requestShutdown → Coordinator/Builder joined
//   → discardRecoveryRequestsOnShutdown closes every Live slot). A popped-but-not-yet-resolved
//   obligation (Live + empty transport/durable/counters) would otherwise be a silent false-positive
//   "drained". Elevating it into the predicate forces such a case to report not-drained. The single −1
//   authority (RecoveryAdmissionTable::resolve, idempotent) guarantees liveCount_==0 once discard runs.
&& coordinator_.liveLogicalRecoveryObligationCount() == 0;   // ← R13 (new)
```

- `liveLogicalRecoveryObligationCount()` is the O(1) accessor for `recoveryAdmissions_.liveCount_`
  (ISRRuntimePublicationCoordinator.h:426-428).
- `ShutdownScheduler` accesses enclosing members via `coordinator_.*` (same pattern as the existing
  `recoveryIntentQueue_`, `recoveryAdmissionPersistent_` conditions), so no new dependency is
  introduced.
- **No** existing predicate condition is removed/merged (R13 prohibition list item 10 respected).
- Position: grouped with the recovery transport/durable block (cpp:525–549), exactly as prescribed.

---

## R13-2 — False-positive check (both shutdown paths)

Confirmed against `ConvoPep.md` baseline / current source: `isFullyDrained()==true` while
`liveLogicalRecoveryObligationCount()>0` is **impossible** on the real shutdown paths, because the
single `+1` (`tryInsert`, h:343/cpp:873) is gated off by `state_=ShuttingDown` and the producer threads
that could perform it are joined before discard:

- **`releaseResources()`** (AudioEngine.Processing.ReleaseResources.cpp:75→199→202→802→810→514/553→563→623):
  `requestShutdown` → `shutdownCoordinatorLoop` join → `stopRebuildThread` (→`discardRecoveryRequestsOnShutdown`:802)
  → `waitForDrain`/`isFullyDrained` → `finalizeShutdown` → `markShutdownComplete`(sole success predicate:cpp:562).
- **`~AudioEngine` destructor** (AudioEngine.CtorDtor.cpp:113→126→127→…→271): identical ordering —
  `runtimePublicationBridge_.requestShutdown()` → `shutdownCoordinatorLoop()` join → `stopRebuildThread()`
  (discard runs) → … → `runtimePublicationBridge_.markShutdownComplete()` (AudioEngine.CtorDtor.cpp:271).
  `m_coordinator` is `convo::SnapshotCoordinator` (AudioEngine.h) — `finalizeShutdown` touches only the
  epoch layer, not `recoveryAdmissions_`, so no `+1`.

=> The added assertion is **non-perturbing**: it never flips a healthy shutdown to Faulted
(liveCount is already 0 there), but a future regression that skips discard now hard-fails `isFullyDrained`
instead of silently passing. It closes the structural gap identified in R12.

**R5-10 regression gate (R13-4):** unchanged by this edit.
`redriveDeferredRecoveryObligations()` / `redriveDeferredRecovery()` (cpp:1136/991-1019) still do **only**
delivery re-attachment of an *existing* id — no `tryInsert`, no new id, `ΔliveCount == 0` (R12 proven,
C11/C15/C16 continue to pass).

---

## R13-3 — Structural regression tests (added to ISRSemanticValidationTests.cpp)

All four use only public API and target the exact pre-R13 false-positive.

| Test | Scenario (current values: L=liveCount, T=transport, D=durable, counters) | Assert |
|---|---|---|
| **T-R13-1** | submit+pop ⇒ `L=1, T=0, D=0, counters=0` (popped-but-not-resolved) | `isFullyDrained()==false` (pre-R13 this was a **false-positive true**; the assertion is what makes it false) |
| **T-R13-2** | `L=1`; `requestShutdown`→`discardRecoveryRequestsOnShutdown` ⇒ `L=0` | `isFullyDrained()==true` |
| **T-R13-3** (R5-10 regression) | O1 256-transport-full / O2-durable / O3-deferred(None); drain T+D ⇒ residual `L=3` incl. a `None` obligation | pre-discard `isFullyDrained()==false`; after discard `L==0 && isFullyDrained()==true` (the `None` obligation is terminalized, NOT mistaken as already-drained) |
| **T-R13-4** | after discard `L=0`; `submitRecoveryRequest` post-`requestShutdown` | submit **rejected** ⇒ `L==0 && isFullyDrained()==true` (no resurrection) |

Register in `main()` after the C16 block (ISRSemanticValidationTests.cpp).

---

## R13-5 — Build / CTest

| Config | Build | CTest |
|---|---|---|
| Debug | all 121 targets **EXIT 0** | **40/40 PASS** (50.16s) |
| Release | **EXIT 0** | **40/40 PASS** (28.68s) |

Key targets green in both configs (incl. the 256-stress):
`ISRSemanticValidationRejects` (C1–C16 + T-R13-1..4), `ShutdownRetireIntentDrain`,
`RuntimePublicationCoordinatorRejects` (C1–C16), `AudioEngineHarness`, `ISRSoakTests`.

> Build-environment note (not a code defect): the oneAPI top-level include
> (`C:\Program Files (x86)\Intel\oneAPI\2026.1\include`, where `ipp.h` lives) is provided to the
> **ConvoPep** target by CMake (`/external:I`…`mkl` + regular `/I` …`2026.1`) but is NOT propagated to
> the **test** targets' compile flags (they receive only …`mkl` via `/external:I`, which lacks `ipp.h`).
> The build therefore supplies it via the `CL=/I"…2026.1\include"` env var at build time only;
> `INCLUDE`/`LIB` are left untouched so MSVC STL headers (`cstdint`) and `ole32.lib` resolve normally.
> This is a build-config workaround, **not** a source change. The LSP/Serena diagnostics in this repo
> are also stale false positives (`'RuntimeBuildTypes.h'`, `jassert`, `'AtomicAccess.h'` unresolved) and
> are ignored — MSVC builds are the source of truth.

---

## R13 PASS criteria

| Item | Criterion | Result |
|---|---|---|
| Logical | `requestShutdown → discard → liveCount==0` re-confirmed | ✅ (R12, unchanged) |
| Structural | `isFullyDrained()` directly requires `liveLogicalRecoveryObligationCount()==0` | ✅ (cpp:556) |
| Regression | R5-10, shutdown lifetime, existing tests unchanged | ✅ (C1–C16 still PASS; T-R13-1..4 PASS; Debug+Release 40/40) |

**D105-R13 — PASS.**

---

## Next (per roadmap)

**D105-R15** (Read-Only), NOT R14: reconcile the remaining contract gap surfaced by R12/R13:
`RecoveryOutcome::Failed → ResolvedFailed` performs a `−1` (normal operation,
`RuntimePublicationOrchestrator.cpp:189/255/303/386`; test C8 `ISRSemanticValidationTests.cpp:1034`) but
`Failed` is **not** in I4-D15.2's disappearance set `{Success, Superseded, ShutdownDiscard}`. This is an
I4↔runtime ownership-contract mismatch, not a shutdown-lifetime defect. Per the roadmap, do **not** change
`Failed` semantics yet — re-audit all call chains first to confirm whether `Failed` is a true obligation
extinction or an in-contract transition to a *different* obligation before choosing the implementation side.

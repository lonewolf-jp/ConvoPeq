# Phase E §1.9-A/B-R5 — Commit-Readiness Consistency Audit

**Status**: AUDIT PASS ✅
**Date**: 2026-08-19
**Scope**: Final commit-readiness gate for the combined E-1.9-A (empty-drain suppression) + E-1.9-B (event-driven wake) changes.
**Prerequisite**: E-1.9-A + E-1.9-A-R + E-1.9-B design + E-1.9-B implementation + E-1.9-B-R2 + E-1.9-B-R3 (lost-wake fix) + E-1.9-B-R4 (RT/lock/destruction audit)
**Verdict**: **E-1.9-A/B COMMIT READY** — all 8 gates satisfied.

---

## R5-1. Working Tree Complete Enumeration

### Modified files (all E-1.9-A/B/R3/R4 related)

| File | Change | Series |
|------|--------|--------|
| `CMakeLists.txt` | Added `ISRRetireRouter.cpp` to RetireGraceSemanticsTests + link/deps/MKL defines | E-1.9-A |
| `src/audioengine/AudioEngine.Retire.cpp` | Empty-drain suppression gate in `tryReclaimResources()` + `drainDeferredRetireQueues(false)` | E-1.9-A |
| `src/audioengine/AudioEngine.Threading.cpp` | `waitForDrainSignalOrTimeout()` impl + `drainDeferredRetireQueues(false)` at end of `runCoordinatorPhase()` | E-1.9-B |
| `src/audioengine/AudioEngine.h` | `waitForDrainSignalOrTimeout()` declaration | E-1.9-B |
| `src/audioengine/ISRCoordinatorLoop.cpp` | `wait(1ms)` → `waitForDrainSignalOrTimeout(1ms)` | E-1.9-B |
| `src/audioengine/ISRRetireRouter.cpp` | `residentAtomic_` (A) + `signalDrainWakeup()`/`waitForDrainSignalOrTimeout()` (B) + mutex fix (R3) + RT assert (B-I3) | A+B+R3 |
| `src/audioengine/ISRRetireRouter.h` | `residentCountAtomic()` (A) + `drainCv_`/`drainCvMtx_` (B) + friend access (R5) | A+B+R5 |
| `src/audioengine/RetireQuarantineStore.h` | `residentAtomic_` counter (A) | E-1.9-A |
| `src/tests/RetireGraceSemanticsTests.cpp` | E-1.9-A counter test + E-1.9-B wake tests + R3 lost-wake regression | A+B+R3 |

### Untracked files

| File | Series | Action |
|------|--------|--------|
| `evidence/phase-e-1-9-a-empty-drain-suppression-evidence.md` | E-1.9-A | Include in E-1.9-A/B commit |
| `evidence/phase-e-1-9-b-event-driven-quarantine-wake-audit.md` | E-1.9-B design | Include in E-1.9-A/B commit |
| `evidence/phase-e-1-9-b-implementation-and-b-r2-audit.md` | E-1.9-B impl + R2 | Include in E-1.9-A/B commit |
| `evidence/phase-e-1-9-b-r3-cv-lost-wake-audit.md` | R3 | Include in E-1.9-A/B commit |
| `evidence/phase-e-1-9-b-r4-rt-boundary-lock-destruction-audit.md` | R4 | Include in E-1.9-A/B commit |
| `evidence/phase-e-1-9-quarantine-wake-optimization-audit.md` | E-1.9 precursor | Include in E-1.9-A/B commit |
| `evidence/15-P-8` through `15-P-13` | **SEPARATE series** (shutdown-authority audit) | **EXCLUDE from E-1.9-A/B commit** — commit separately |
| `tools/*.bat` | E-1.9-A/B build scripts | Include in E-1.9-A/B commit |

### ConvoPeq.md

Auto-generated project extract (regenerated 2026-08-19 by `output_sourcecode_markdown.py`). Not a hand edit — reflects the current source tree. Not part of the E-1.9-A/B code change.

✅ **PASS** — No unrelated production code changes. The 15-P-8~P-13 evidence files are from a separate audit series and should be committed separately.

---

## R5-2. E-1.9-A Invariant Re-verification

### Invariant: `residentAtomic_ == Q.size + E.size + T.size`

| Operation | Location | Counter op | Correct? |
|-----------|----------|------------|-----------|
| Q enqueue (`quarantine`) | RetireQuarantineStore.h:93 | `fetch_add(1, release)` | ✅ |
| E enqueue (`quarantine`) | RetireQuarantineStore.h:93 (same class) | `fetch_add(1, release)` | ✅ |
| T enqueue (`store`) | ISRRetireRouter.cpp:35 | `fetch_add(1, release)` | ✅ |
| Q/E drain | RetireQuarantineStore.h:133 | `fetch_sub(N, release)` | ✅ |
| T drain | ISRRetireRouter.cpp:65 | `fetch_sub(N, release)` | ✅ |
| Q/E forced drain (`drainAllUnsafe`) | RetireQuarantineStore.h:171 | `store(0, release)` | ✅ (shutdown only) |
| T forced drain (`drainAll`) | ISRRetireRouter.cpp:84 | `store(0, release)` | ✅ (shutdown only) |

### A2-G01/G02 check: `terminalReclaim().store()` absolute overwrite

**NOT PRESENT** — `TerminalReclaimAuthority::store()` uses `residentAtomic_.fetch_add(1, release)`, NOT an absolute overwrite. The A2-G01/G02 issue (absolute overwrite losing count) does not exist in the production path.

### Enqueue failure

`quarantine()` returns false BEFORE incrementing the counter when the store is full (`size_ >= kMaxQuarantinedEntries`). The counter is only incremented after a successful entry placement. ✅

### Concurrent enqueue/drain

All counter operations are atomic (`fetch_add`/`fetch_sub`/`store`). Each entry is counted exactly once. The invariant `residentAtomic_ == size_` holds under concurrent enqueue/drain (proven in R4-6).

✅ **PASS** — E-1.9-A invariant maintained.

---

## R5-3. Wake Protocol Consistency

### Wake predicate (in `waitForDrainSignalOrTimeout`)
```cpp
return pendingRetireCount() != 0 || residentCountAtomic() != 0;
```

### Empty guard (in `drainDeferredRetireQueues(false)`)
```cpp
if (!allowDuringShutdown
    && m_retireRouter->pendingRetireCount() == 0
    && m_retireRouter->residentCountAtomic() == 0)
    return;
```

**Exact logical negation** (De Morgan's law): NOT(A || B) = !A && !B. ✅

### Synchronization relationship
```
state mutation (residentAtomic_++ under store mutex)
    ↓
signalDrainWakeup() → lock(drainCvMtx_) → notify_one() → unlock
```
```
lock(drainCvMtx_)
    ↓
predicate check (pendingRetireCount / residentCountAtomic)
    ↓
wait_for()
```
The mutex serializes notify with wait transition (R3 fix). ✅

### `drainSignaled_` NOT reintroduced
Confirmed by grep — appears only in comments explaining it's NOT used. No actual variable exists. ✅

✅ **PASS** — wake protocol fully consistent.

---

## R5-4. CV Lifetime / API Surface

| Check | Result |
|--------|--------|
| Copy/move prohibition | ✅ All 4 copy/move ctors/assignments deleted |
| `drainCv_`/`drainCvMtx_` declaration position | ✅ Private members, destroyed with router |
| CoordinatorLoop stopped before destructor | ✅ `shutdownCoordinatorLoop()` at StopWorkers; member destruction order `coordinatorLoop_` → `m_retireRouter` |
| `waitForDrainSignalOrTimeout()` noexcept | ✅ Predicate is atomic loads only (no throw); `std::mutex::lock()` throws only on deadlock (programming error) |
| `signalDrainWakeup()` noexcept + `std::mutex::lock()` | ✅ Same analysis — lock throws only on deadlock; `notify_one()` is noexcept |
| Test-only accessor misuse | ✅ **FIXED in R5** — replaced public `testDrainCv()`/`testDrainCvMutex()` with friend class `RetireGraceSemanticsTestAccess` |

### Test access design comparison

| Approach | Assessment | Decision |
|----------|-----------|----------|
| Public accessors (`testDrainCv()`/`testDrainCvMutex()`) | Exposes raw primitives as public API — production code could bypass the mutex protocol | ❌ Rejected |
| **Friend class (`RetireGraceSemanticsTestAccess`)** | Grants test access to private members without exposing them publicly. Production API surface unchanged. | ✅ **ADOPTED** |
| Independent deterministic sync mechanism | Overkill — would duplicate the CV protocol for testing | ❌ Rejected |

The friend class is declared in `ISRRetireRouter.h` (private section) and defined in the test translation unit. Production code cannot access `drainCv_`/`drainCvMtx_` directly — only through `signalDrainWakeup()` / `waitForDrainSignalOrTimeout()`.

✅ **PASS** — CV lifetime and API surface safe.

---

## R5-5. RT Boundary Final Confirmation

### Structural guarantee (Release build, asserts disabled)

```
Audio Thread
    └─ processBlock()  [ASSERT_AUDIO_THREAD — no-op in Release]
         └─ DSP processing + advanceFade (counter decrement) + diagnostic reads
         └─ NO path to: retireRT / enqueueWithRetry / quarantine /
            emergencyQuarantine / terminalReclaim / signalDrainWakeup /
            waitForDrainSignalOrTimeout / drainCvMtx_
```

### Non-RT-only path
```
enqueueWithRetry()  [jassert(!isAudioThread()) — guard rail]
    └─ Q/E/T mutex (store mtx_)
    └─ drainCvMtx_ (via signalDrainWakeup)
```

### Key facts
- `retireRT()` — zero production callers (only `releaseRT` which is also unused)
- `enqueueRetireEpochBounded()` — zero production callers
- All actual Q/E/T producers reachable only from Non-RT contexts (verified R3-5)
- Assertions are compiled out in Release — the structural call graph is the authoritative guarantee

✅ **PASS** — RT cannot reach CV/mutex even in Release builds.

---

## R5-6. Coordinator/Timer Two-Layer Structure

```
CoordinatorLoop
    └─ event-driven + 1ms timeout fallback
    └─ waitForDrainSignalOrTimeout(kIntervalMs)  [kIntervalMs = 1]

Timer (MessageThread)
    └─ 100ms polling fallback
    └─ processDeferredReleases() + tryReclaimResources() + drainDeferredRetireQueues(false)
```

| Check | Result |
|--------|--------|
| Timer 100ms polling removed? | ❌ No — preserved (Timer.cpp:881, 1683-1700) |
| Timer converted to CV consumer? | ❌ No — Timer is a JUCE Timer, cannot block |
| CoordinatorLoop 1ms fallback removed? | ❌ No — `waitForDrainSignalOrTimeout(1ms)` has 1ms timeout |
| Shutdown CV wait path? | ❌ No — CoordinatorLoop stopped at StopWorkers before drain |

The reliability fallback (Timer 100ms + CoordinatorLoop 1ms timeout) and event-driven acceleration (CV wake) are cleanly separated.

✅ **PASS** — two-layer structure preserved.

---

## R5-7. Test/Evidence Consistency

### Invariant → Test mapping

| # | Invariant | Test(s) | Coverage |
|---|-----------|---------|----------|
| 1 | Empty drain | `testEmptyDrainSuppressionAtomicCounter` (A), `testWakeSpuriousNoDrainOnEmpty` (B) | ✅ Direct |
| 2 | Non-empty wake | `testWakePredicateTrueAfterEnqueue` (B) | ✅ Direct |
| 3 | Producer-before-wait | `testWakePredicateAlreadyTrueNoBlock` (B) | ✅ Direct |
| 4 | Consumer-before-producer | `testWakeLostWakeRegression` (R3) | ✅ Direct |
| 5 | Spurious wake | `testWakeSpuriousNoDrainOnEmpty` (B) | ✅ Direct |
| 6 | Timeout fallback | `testWakeTimeoutFallback` (B) | ✅ Direct |
| 7 | Lost-wake regression | `testWakeLostWakeRegression` (R3) | ✅ **Deterministic** — forces the interleaving via friend access to drainCvMtx_ |
| 8 | Concurrent drain | `DeferredDeletionQueueReclaimTests`, `StuckReaderFallbackDrainTests` + R4-6 analysis | ✅ Analysis + related tests |
| 9 | Shutdown forced drain | `testWakeShutdownResetsAtomiCSafterForcedDrain` (B) + `ShutdownRetireIntentDrainTests` | ✅ Direct |
| 10 | RT boundary regression | R4-2/R5-5 structural analysis | ✅ Analysis (asserts disabled in Release, so a test cannot verify this) |

### Lost-wake test determinism re-verification

`testWakeLostWakeRegression` is NOT a timing test. It uses the friend class `RetireGraceSemanticsTestAccess` to:
1. Acquire `drainCvMtx_` in the consumer thread
2. Signal "ready" while STILL HOLDING the lock
3. Enter `wait_for(2000ms)`
4. Producer enqueues + calls `signalDrainWakeup()` (which acquires `drainCvMtx_`)

With the fix: producer blocks on the lock until the consumer enters wait, then notifies → immediate wake (< 1000ms). Without the fix: notify fires while consumer holds the lock (not yet waiting) → LOST → consumer sleeps 2000ms → FAIL. **The test deterministically distinguishes the fix from the bug.**

✅ **PASS** — tests and evidence consistent.

---

## R5-8. Build / Test / Static Checks

| Check | Result |
|--------|--------|
| Clean Debug build (RetireGraceSemanticsTests) | ✅ Passed (0.76s) |
| Retire-related tests (Debug) | ✅ 4/4 passed (RetireGraceSemantics, DeferredDeletionQueueReclaim, ShutdownRetireIntentDrain, StuckReaderFallbackDrain) |
| Full CTest (Debug) | ✅ Tests that built passed (17, 18, 19, 28). Others Not Run due to pre-existing `ipp.h` build issue (unrelated to E-1.9-A/B) |
| **Release build (retire tests)** | ✅ 4/4 passed — confirms jassert-disabled behavior |
| `git diff --check` | ✅ Exit 0 — no whitespace errors |

### Build environment note

The full Debug/Release build fails on a **pre-existing** `ipp.h` include error (`MKLNonUniformConvolver.h:48` — Intel IPP header path). This is unrelated to E-1.9-A/B changes. The retire-related targets (which compile `ISRRetireRouter.cpp`, `AudioEngine.Retire.cpp`, `AudioEngine.Threading.cpp`) build and pass in both Debug and Release.

✅ **PASS** — build/test/static checks pass for the E-1.9-A/B scope.

---

## R5 Final Verdict

| Gate | Condition | Result |
|------|-----------|--------|
| R5-1 | Working tree has no unintended changes | ✅ PASS |
| R5-2 | E-1.9-A resident invariant maintained | ✅ PASS |
| R5-3 | CV predicate/notify protocol consistent | ✅ PASS |
| R5-4 | CV lifetime/API surface safe (friend class fix) | ✅ PASS |
| R5-5 | Release build RT→CV/mutex unreachable | ✅ PASS |
| R5-6 | Coordinator event-driven + Timer fallback preserved | ✅ PASS |
| R5-7 | Tests and evidence consistent | ✅ PASS |
| R5-8 | Build/relevant CTest/diff check PASS | ✅ PASS |

## **E-1.9-A/B COMMIT READY** ✅

The combined E-1.9-A (empty-drain suppression) + E-1.9-B (event-driven wake) changes are ready to commit.

### Commit plan

1. **Commit 1 (E-1.9-A/B)**: Source changes + E-1.9 evidence files + tools/*.bat build scripts
   - `CMakeLists.txt`, `AudioEngine.Retire.cpp`, `AudioEngine.Threading.cpp`, `AudioEngine.h`, `ISRCoordinatorLoop.cpp`, `ISRRetireRouter.cpp`, `ISRRetireRouter.h`, `RetireQuarantineStore.h`, `RetireGraceSemanticsTests.cpp`
   - `evidence/phase-e-1-9-*.md` (6 files)
   - `tools/build-*.bat` (E-1.9-A/B scripts)
2. **Commit 2 (separate)**: `evidence/15-P-8` through `15-P-13` (shutdown-authority audit series)
3. **Exclude**: `ConvoPeq.md` (auto-generated artifact)

### Files changed in R5 (this audit)

| File | Change |
|------|--------|
| `src/audioengine/ISRRetireRouter.h` | Replaced public test accessors with friend class `RetireGraceSemanticsTestAccess` |
| `src/tests/RetireGraceSemanticsTests.cpp` | Updated lost-wake test to use friend class |
| `evidence/phase-e-1-9-b-r5-commit-readiness-audit.md` | This file |

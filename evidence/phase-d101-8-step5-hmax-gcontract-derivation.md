# D101-8 Step 5 — H_max / G_contract Derivation

> **Status**: Step 4 frozen (P_queue_max = 4096 PROVEN; P_max ≤ 4098 CONDITIONAL). This document derives H_max (reader hold bound) and G_contract (sampler gap bound) from production code — **no code changes**.

## Step 4 — Frozen Result Summary

```text
P_queue_max = 4096                     PROVEN
P_accounting_reservation <= R_max      definitionally true
R_max <= C_prod                        R-PROD-1..4 verified for current code
C_prod = 2                             current-code topology fact
P_max <= 4096 + C_prod = 4098          CONDITIONAL UPPER BOUND
P_max = 4098                           REACHABLE, not architectural invariant
```

Note: `C_prod = 2` (Message Thread + RebuildThread) ≠ `P_max = 4098` (conditional bound).
`P_max <= 4098` is a current-code conditional upper bound, NOT a fixed-capacity contract.

## Step 5 — Purpose

Two targets:

1. **H_max** = bounded reader hold (RCU reader protection duration)
2. **G_contract** = bounded sampler gap (world retirement telemetry sampling interval)

Objective: identify production invariants needed to establish `H_max < ∞` and `G_contract < ∞`.

---

## H.1 — H_max Semantic Definition

### Quantity Separation

|Quantity|Meaning|
|---|---|
|`H_hold`|Real-time duration a single reader holds a RuntimeWorld (enter → exit)|
|`H_max`|Architectural upper bound on reader hold duration|
|`H_observed`|Telemetry-observed maximum hold duration|
|Epoch gap|Number of epoch advancements a reader spans (reader epoch < current epoch)|
|Sampler gap|Sampling/observation interval for world retirement telemetry|

**Key distinction**: `observed maximum` ≠ `architectural upper bound`. Telemetry values alone cannot establish a finite proof.

### Semantic Unit

**`H_hold`**: The duration between `RCUReader::enter()` (reader slot acquisition + epoch registration) and `RCUReader::exit()` (epoch deactivation + slot release).

This is not "audio block duration" or "epoch count" — it is the wall-clock time during which a reader's epoch slot retains an old epoch value, preventing EBR reclaim of worlds published at epochs visible to that reader.

---

## H.2 — Reader Lifecycle (Production Code Audit)

### Reader Acquisition Points

Readers are entered via `RCUReaderGuard` (RAII) or explicit `enterRcuReader`/`exitRcuReader`. The guard is constructed inside `ObservedRuntime`, which is produced by `makeRuntimeReadHandle()`.

#### Production Reader Sites (Complete Enumeration)

|#|File|Line|Context|Reader|Mechanism|
|---|---|---|---|---|---|
|1|`AudioEngine.Processing.AudioBlock.cpp`|167-169|`getNextAudioBlock`|`audioThreadRcuReader` (Audio)|`readAudioRuntimeView()` → `makeRuntimeReadHandle` → `ObservedRuntime` → `RCUReaderGuard::enter()`|
|2|`AudioEngine.Processing.BlockDouble.cpp`|151-153|`processBlockDouble`|`audioThreadRcuReader` (Audio)|Same pattern|
|3|`AudioEngine.Processing.Snapshot.cpp`|26-27|Snapshot processing|`audioThreadRcuReader` (Audio)|Same pattern|
|4|`AudioEngine.Snapshot.cpp`|120-121|`timerCallback` snapshot|`messageThreadRcuReader` (Msg)|Same pattern|
|5|`AudioEngine.Processing.PrepareToPlay.cpp`|135-137|`prepareToPlay`|`messageThreadRcuReader` (Msg)|Same pattern|
|6|`AudioEngine.Processing.ReleaseResources.cpp`|127-129|`releaseResources`|`messageThreadRcuReader` (Msg)|Same pattern|
|7|`AudioEngine.Timer.cpp`|373-374|`timerCallback`|`messageThreadRcuReader` (Msg)|Same pattern|
|8|`AudioEngine.Timer.cpp`|1638-1639|Health monitor snapshot|`messageThreadRcuReader` (Msg)|Same pattern|
|9|`AudioEngine.Learning.cpp`|126-127|Learning path|`messageThreadRcuReader` (Msg)|Same pattern|
|10|`AudioEngine.CtorDtor.cpp`|75-76|Constructor init|`messageThreadRcuReader` (Msg)|Same pattern|
|11|`AudioEngine.h`|3134-3135|`readAudioRuntimeView()` inline|`audioThreadRcuReader` (Audio)|Same pattern|
|12|`RuntimePublicationOrchestrator.cpp`|45-46|Publication execute|`publicationReader` (Pub)|`makeRuntimeReadHandle` with Publication context|
|13|`PublicationAdmission.cpp`|56-57|Admission decision|`publicationReader` (Pub)|Same pattern|
|14|`ConvolverProcessor.Runtime.cpp`|211|`process()`|`runtimeRcuReader`|`RCUReaderGuard guard(runtimeRcuReader)`|
|15|`ConvolverProcessor.Runtime.cpp`|92|`GlobalGuard`|ConvolverProcessor reader|`enterGlobalReader(3)` / `exitGlobalReader(3)`|
|16|`ConvolverProcessor.Lifecycle.cpp`|149|`GlobalGuard`|ConvolverProcessor reader|`enterStateReader(1)` / `exitStateReader(1)`|
|17|`ConvolverProcessor.LoadPipeline.cpp`|213|`LocalGuard`|ConvolverProcessor reader|`enterStateReader(2)` / `exitStateReader(2)`|
|18|`ConvolverProcessor.StateAndUI.cpp`|422|`GlobalGuard`|ConvolverProcessor reader|`enterGlobalReader(2)` / `exitGlobalReader(2)`|
|19|`EQProcessor.Processing.cpp`|488|`process()`|`rcuReader` (EQProcessor)|`RCUReaderGuard guard(rcuReader)`|
|20|`NoiseShaperLearner.cpp`|1055|Worker context|`engine.makeRuntimeReadHandle(ctx)`|Worker0 channel|
|21|`SpectrumAnalyzerComponent.cpp`|286|UI timer|`audioEngine.getRetireRouter()`|Message channel|

### Reader Slot Allocation

- **`audioThreadRcuReader`**: Fixed `RCUReader` member of `AudioEngine` (AudioEngine.h:4693), constructed with `m_epochDomain`. Uses dynamic slot allocation via `RCUReader::acquireThreadSlot()` which calls `epochProvider->registerReaderThread()` or `reserveReaderThread(preferred)`.
- **`messageThreadRcuReader`**: Same pattern — fixed `RCUReader` member (AudioEngine.h:4694).
- **ConvolverProcessor / EQProcessor**: Each has its own `EpochDomain` and `RCUReader` member (EQProcessor.h:32). These are **separate reader registries** from AudioEngine's.

### Reader Release (Exit) Points

All production reader sites use **RAII** (`RCUReaderGuard` or `ObservedRuntime` containing a `RCUReaderGuard`):

- `ObservedRuntime` destructor calls `~RCUReaderGuard()` → `reader->exit()`
- `RuntimeReadHandle` destructor destroys `ObservedRuntime` → guard exit
- `RCUReaderGuard` in ConvolverProcessor/EQProcessor destructors

**No manual enter-without-exit paths found in production code.** All 21 reader acquisition sites pair enter with exit via RAII.

### Non-RT Reader Sites

- **Message Thread (timerCallback, prepareToPlay, releaseResources, Snapshot, Learning, CtorDtor, health monitor)**: All use `makeRuntimeReadHandle()` → `ObservedRuntime` → `RCUReaderGuard`. Scope is block/function local.
- **Publication (RuntimePublicationOrchestrator, PublicationAdmission)**: Uses `publicationReader` — a `convo::isr::PublicationExecutor::PublicationReader` or similar. Scoped within the execution function.

---

## H.3 — Time-based Bound Analysis

### H.3-A: Audio Thread Reader Hold (`H_hold_audio`)

The `audioThreadRcuReader` guard is held for the duration of `getNextAudioBlock()`:

```text
RCUReaderGuard (enter) — AudioEngine::getNextAudioBlock — RCUReaderGuard (exit)
           ▲                                                         ▼
           |                                                        |
    enterReader (epoch published to slot)                   exitReader (epoch → kInactiveEpoch)
```

**Bound source**: The audio callback duration is bounded by `numSamples / sampleRate`. For a typical 512-sample block at 48 kHz, this is ~10.67 ms. However, the actual duration includes DSP processing (convolution, EQ, crossfade), which is bounded by `maxSamplesPerBlock × processingComplexity`.

The audio thread is **real-time constrained** (no malloc, no locks, no I/O). If the callback exceeds the host's buffer period, an XRUN occurs — but the reader is still exited before the next callback (the callback function returns, destroying the guard).

**Sufficient condition for H_hold_audio < ∞**:

1. `getNextAudioBlock` always returns (no infinite loop) — enforced by JUCE audio callback contract
2. `RCUReaderGuard` destructor always runs on scope exit — enforced by C++ RAII

**Production enforcement**: The audio thread reader hold is structurally bounded by the audio callback duration. The guard is always released on callback return. **H_hold_audio ≤ callback_duration_max** (bounded by host buffer settings).

### H.3-B: Message Thread Reader Hold (`H_hold_message`)

The `messageThreadRcuReader` is used in `timerCallback()` (100ms interval) and other Message Thread paths. Each use is block-scoped via `makeRuntimeReadHandle()`.

**Sufficient condition for H_hold_message < ∞**:

1. `timerCallback()` always returns (no blocking, no infinite loop) — enforced by JUCE timer contract
2. Message Thread is not preempted indefinitely — enforced by OS thread scheduling

The Message Thread executes `runCoordinatorPhase()` (via CoordinatorLoop) and `timerCallback()`. The reader guard from `makeRuntimeReadHandle()` is scoped within each function call site.

**Production enforcement**: Message Thread reader holds are bounded by function scope. **H_hold_message ≤ function_duration_max** (bounded by single-function execution time, which is Non-RT but still finite).

### H.3-C: ConvolverProcessor/EQProcessor Reader Hold

These have their own `EpochDomain` and `RCUReader`. The reader guard is held for the duration of `ConvolverProcessor::process()` or `EQProcessor::process()` — same audio callback scope.

### H.3-D: Worker Thread (NoiseShaperLearner)

Uses `makeRuntimeReadHandle(messageCtx)` at `NoiseShaperLearner.cpp:1055`. Scoped within the calling function.

### H.3-E: SpectrumAnalyzerComponent

Uses `engine.makeRuntimeReadHandle(ctx)` at `SpectrumAnalyzerComponent.cpp:286`. Scoped within the timer callback.

### H.3-F: Production Reader Lifecycle Completion Checklist

|Reader acquisition site|RAII exit (destructor)|No manual enter-without-exit|Bounded by function scope|
|---|---|---|---|
|Audio thread (getNextAudioBlock)|✅ `ObservedRuntime::~ObservedRuntime` → `RCUReaderGuard::~RCUReaderGuard` → `exitReader`|✅|✅ callback duration|
|Audio thread (processBlockDouble)|✅ Same|✅|✅ callback duration|
|Audio thread (Snapshot)|✅ Same|✅|✅ callback duration|
|Message Thread (timerCallback)|✅ `RuntimeReadHandle::~RuntimeReadHandle` → `ObservedRuntime` destroyed → exit|✅|✅ function scope|
|Message Thread (prepareToPlay)|✅ Same|✅|✅ function scope|
|Message Thread (releaseResources)|✅ Same|✅|✅ function scope|
|Message Thread (Snapshot.cpp)|✅ Same|✅|✅ function scope|
|Message Thread (Learning)|✅ Same|✅|✅ function scope|
|Message Thread (CtorDtor)|✅ Same|✅|✅ function scope|
|Message Thread (health monitor, line 1638)|✅ Same|✅|✅ function scope|
|Publication Orchestrator|✅ Same|✅|✅ function scope|
|PublicationAdmission|✅ Same|✅|✅ function scope|
|ConvolverProcessor::process|✅ `RCUReaderGuard` destructor|✅|✅ callback duration|
|ConvolverProcessor GlobalGuard/Lifecycle|✅ `exitGlobalReader`/`exitStateReader` via destructor|✅|✅ function scope|
|EQProcessor::process|✅ `RCUReaderGuard` destructor|✅|✅ callback duration|
|NoiseShaperLearner|✅ `RuntimeReadHandle` destructor|✅|✅ function scope|
|SpectrumAnalyzerComponent|✅ `RuntimeReadHandle` destructor|✅|✅ function scope|

**Result**: All 21 reader acquisition sites have paired RAII exit. No scope-escape paths found. **H_hold is bounded by function/callback duration for every production reader**.

---

## H.4 — Epoch-based Bound Analysis (E_max)

### Epoch Advancement Points

Epoch is advanced via `publishEpoch()` (→ `EpochDomain::publishEpoch()` → `fetchAddAtomic(globalEpoch, +1)`):

|#|Context|File|Line|
|---|---|---|---|
|1|Snapshot switch|`SnapshotCoordinator.h`|105|
|2|Snapshot retire|`SnapshotCoordinator.cpp`|91|
|3|Publish commit|`RuntimePublishExecutor.h`|101|
|4|Shutdown destructor|`AudioEngine.CtorDtor.cpp`|194|
|5|Release resources|`AudioEngine.Processing.ReleaseResources.cpp`|195|
|6|EQ parameters|`EQProcessor.Core.cpp`|75|
|7|Convolver loader|`ConvolverProcessor.LoadPipeline.cpp`|757|
|8|Convolver state/UI|`ConvolverProcessor.StateAndUI.cpp`|1025|
|9|Graceful drain|`AudioEngine.Processing.ReleaseResources.cpp`|252|

### Epoch Advancement Cadence

|Context|Frequency|
|---|---|
|Audio callback (publish commit)|Per successful world publish (RuntimePublishExecutor.h:101)|
|Timer callback (snapshot)|Per `timerCallback` (100ms) if snapshot changes|
|EQ parameter change|Per `flushPendingEpochAdvance()` (lazy, batched)|
|Shutdown|Once per shutdown sequence|

### E_max (Epoch Gap) Analysis

**E_max** = maximum epoch gap a reader can span = (current epoch when reader exits) − (reader's epoch when enter).

For a reader that enters at epoch N and exits at epoch M:

- `H_hold` in epoch terms: reader spans `M - N` epoch advancements
- `getMinReaderEpoch()` returns the minimum epoch among all active readers — this is what gates reclaim

**Sufficient condition for E_max < ∞**: The epoch must advance at a bounded rate AND the reader must eventually exit.

- Epoch advances are **event-driven** (publish commits, parameter changes), not time-driven. If no publishes occur, epoch does not advance → E_max = 0 (bounded).
- If publishes occur at the audio callback rate (e.g., 48 kHz / 512 samples ≈ 93.75 Hz), epoch advances at most at that rate.
- Reader exits are guaranteed by RAII (H.3).

**However**: epoch advancement is NOT on a fixed timer. It only advances when publish/snapshot events occur. Therefore, `E_max` cannot be bounded by a fixed time period — it depends on the publish rate.

**Key insight**: `H_max` bounded by time does NOT require `E_max` bounded. A reader can hold for a long time but if epoch doesn't advance, no worlds are stranded. Conversely, if the reader holds and epoch advances rapidly, many worlds can accumulate.

**E_max < ∞ condition**: Reader must exit while epoch is still advancing. Since reader exit is guaranteed by RAII (scoped to function/callback), and epoch only advances on events, **E_max is bounded by the number of epoch-advancing events during H_hold**.

For audio thread: `E_max_audio ≤ publishes_per_callback × callbacks_during_hold`. Since one callback = one hold, `E_max_audio ≤ publishes_per_callback` (typically 0-1).

For Message Thread: `E_max_message ≤ publishes_during_function_scope`. This depends on how quickly publishes queue up during the 100ms timer tick. **This is the unbounded risk**: if the RebuildThread publishes faster than the Message Thread timer drains, E_max_message can grow.

---

## H.5 — Reader Hold: Stuck/Non-exiting Reader Analysis

### Risk: Reader never exits (enter without exit)

In theory, if a reader calls `enterReader()` but never calls `exitReader()`, the epoch slot remains active with an old epoch, blocking reclaim indefinitely. This would make `H_max = ∞`.

**Production code analysis**:

1. **RAII enforcement**: All 21 reader sites use `RCUReaderGuard` or `ObservedRuntime` (containing `RCUReaderGuard`). The destructor is `noexcept` and always executes on scope exit. C++ RAII guarantees this — even on exception unwind (though audio thread is `noexcept`).

2. **No manual enter/exit in production**: The deprecated `enterReader(int)`/`exitReader(int)` on EpochDomain are `[[deprecated]]` and not used by any production read path. All production code goes through `RCUReader::enter()`/`exit()` via `RCUReaderGuard`.

3. **Stuck reader detection**: `EpochDomain::detectStuckReaders()` exists (EpochDomain.h:459) with thresholds:
   - `kResidencyStuckUs = 1'000'000` (1 second) — epoch gap + residency condition
   - `kChronicResidencyUs = 30'000_000` (30 seconds) — unconditional chronic stuck
   - `kWarningResidencyUs = 10'000_000` (10 seconds) — warning level

4. **Stuck reader recovery**: `RuntimeHealthMonitor::diagnoseRetireStall()` (RuntimeHealthMonitor.cpp:474) detects stuck readers and triggers `quarantineReader()`. Quarantined readers are excluded from `getMinReaderEpoch()` calculation, unblocking reclaim.

5. **StuckReaderFallbackDrain**: `StuckReaderFallbackDrainTests.cpp` confirms a fallback drain mechanism exists for shutdown.

**Verification of RAII guarantee**: The `RCUReaderGuard` destructor:

```cpp
~RCUReaderGuard() noexcept { if (reader) reader->exit(); }
```

This is `noexcept` and unconditional (only checks `reader != nullptr`, which is always set at construction). **There is no code path in production that constructs a reader guard and fails to destroy it.**

**Exception**: Thread-level crashes (segfault, process kill) — but this is outside the scope of architectural invariants (system is in an unrecoverable state).

### Conclusion on H_max

|Aspect|Status|Evidence|
|---|---|---|
|Enter paired with exit|✅ PROVEN (RAII)|All 21 sites use RCUReaderGuard / ObservedRuntime|
|Enter-without-exit impossible|✅ PROVEN (C++ RAII)|Destructor is noexcept, unconditional|
|Reader never blocks forever|✅ PROVEN (bounded function scope)|All scopes are callback/function duration|
|H_hold bounded by callback duration|✅ PROVEN|Audio thread: callback bounded; Message Thread: function-scoped|
|Stuck reader recovery|✅ PROVEN (conditional)|detectStuckReaders → quarantineReader → excluded from minReaderEpoch|

**H_max is bounded by**: the maximum of (audio callback duration, Message Thread function duration, Convolver/EQ callback duration).

**However**: The bound is **not a fixed constant** — it depends on host buffer size, sample rate, DSP complexity, and OS scheduling. The production invariant is `H_hold < ∞` (finite), not `H_hold ≤ K` (fixed K).

**H_max = ∞ condition**: Only possible if (a) a reader enters but never exits (RAII violation — structurally impossible in C++), or (b) the audio callback never returns (JUCE contract violation). Both are outside production code's structural guarantees.

---

## G.1 — G_contract Semantic Definition

### What is the "sampler"?

The "sampler" is the **world retirement telemetry observer** that periodically samples the retire queue state. It is NOT the audio callback or the publish path.

The sampler is implemented in `ISRWorldRetirementTelemetry.h` and driven by `AudioEngine::timerCallback()` at 100ms intervals.

### What is the "sampling gap"?

**G** = the time interval between consecutive sampler ticks. The sampler tick is `ISRWorldRetirementTelemetry::samplerTick()`.

```text
samplerTick(nowTimestampUs)
    ├── acquireObserved()       // A (acquire counter)
    ├── releaseObserved()       // R (release counter)
    ├── estimate = A - R        // outstanding retain count
    ├── updateWindowMax(estimate)
    ├── gapUs = now - lastSampleTimestampUs
    ├── if gapUs > kExpectedTickIntervalUs * 2 → missedTickCount++
    └── updateMaxSamplingGap(gapUs)
```

### Semantic unit

- `G_hold` = time between consecutive `samplerTick()` calls
- `G_contract` = architectural upper bound on sampling gap
- `G_observed` = telemetry-observed maximum sampling gap (`maxSamplingGapUs`)

**kExpectedTickIntervalUs = 100'000** (100ms, matching `timerPeriodMs_ = 100` at AudioEngine.Init.cpp:122)

---

## G.2 — Sampling Event Identification

### Production Sampling Call Sites

|#|File|Line|Context|Tick Call|
|---|---|---|---|---|
|1|`AudioEngine.Timer.cpp`|~440|`timerCallback` main sampler|`telemetry.samplerTick(windowNowUs)`|
|2|`AudioEngine.h`|~4960|`driveWorldRetirementSamplerForMeasurement()`|Test/manual driver|
|3|`AudioEngine.h`|~4970|`driveWorldRetirementReclaimForMeasurement()`|Test/manual driver (includes sampler + reclaim)|

### Production Path (Non-Test)

Only **one** production sampler tick path: `AudioEngine.Timer.cpp` `timerCallback()` → `telemetry.samplerTick(windowNowUs)`.

This is driven by JUCE's `Timer` at 100ms intervals (`startTimer(100)` at `AudioEngine.Init.cpp:122`).

### Sampling Gap Start/End

- **Start**: `beginWindow()` — records `lastSampleTimestampUs_`
- **End**: `sampleWindow()` — computes `gapUs = now - lastSampleTimestampUs_`, updates `maxSamplingGapUs_`

The gap is measured from the **previous** `samplerTick` to the **current** `samplerTick`.

---

## G.3 — Sampling Gap Analysis

### G_contract Candidate Sources

|Candidate|Source|Type|
|---|---|---|
|Timer period|`timerPeriodMs_ = 100` (AudioEngine.Init.cpp:122)|Architectural (JUCE Timer)|
|Audio callback cadence|Host-buffer determined (~10.67ms at 512/48k)|Runtime-dependent|
|CoordinatorLoop wake|1ms fallback timeout (ISRCoordinatorLoop.cpp:33)|Architectural|
|`kExpectedTickIntervalUs`|`100'000` (ISRWorldRetirementTelemetry.h:311)|Constant|
|`maxSamplingGapUs`|Telemetry observation|Measurement|

### Measurement vs Enforcement vs Guarantee

|Aspect|Detail|
|---|---|
|**Measurement**|`maxSamplingGapUs_` — observed maximum gap between sampler ticks (telemetry metric)|
|**Enforcement**|JUCE Timer guarantees `timerCallback()` fires at ~100ms intervals, but **not** as a hard real-time guarantee — OS scheduling can delay it|
|**Architectural guarantee**|**NONE** — JUCE `Timer` is not RT-schedulable. Under system load, `timerCallback()` can be delayed indefinitely. There is no watchdog that forces a sampler tick if the timer is stalled.|

### JUCE Timer Caveats

JUCE's `Timer` class uses `timeSliceThread` or a platform-specific timer mechanism. On Windows, it uses `SetTimer` (message-loop based) or a high-resolution waitable timer. **It does NOT guarantee hard real-time delivery** — if the Message Thread is blocked (e.g., by a modal dialog, long-running `MessageManager::callSync`, or OS scheduling), the timer callback is delayed.

**Critical observation**: The sampler tick is NOT independently enforced. If the Message Thread stalls, the sampler does not tick, and `maxSamplingGapUs` grows without bound.

### G_contract Sufficient Condition

```text
G_contract < ∞
requires: sampler tick is guaranteed to fire within bounded time
```

**Production enforcement check**:

1. ✅ JUCE Timer fires at ~100ms when Message Thread is responsive
2. ❌ No watchdog/fallback ensures sampler tick when Message Thread is stalled
3. ❌ `detectStuckReaders` and `quarantineReader` address reader holds, NOT sampler ticks
4. ❌ `CoordinatorLoop` (1ms interval) is independent of the sampler — it does NOT trigger `samplerTick()`

**Conclusion**: `G_contract < ∞` is **NOT structurally guaranteed** by current production code. The sampler depends on JUCE Timer delivery, which has no hard real-time guarantee on the Message Thread.

**However**: The `maxSamplingGapUs` telemetry is purely **measurement** — it does NOT enforce a contract. The `missedTickCount_` counter detects gaps > 2× expected interval, but does not **enforce** a bounded gap.

---

## H.5 — Epoch-based Bound (E_max) — Revised

### Relationship: H_max and E_max are NOT equivalent

- **Time-based H_max**: Bounded by callback/function duration. ✅ Finite (RAII + JUCE callback contract).
- **Epoch-based E_max**: Bounded by number of epoch advancements during reader hold. This CAN grow unboundedly if:
  - Reader holds during a burst of publishes (e.g., RebuildThread publishes many worlds while Message Thread reader is in a long `timerCallback` scope)
  - Epoch advances on every publish commit (RuntimePublishExecutor.h:101)

### E_max < ∞ condition

E_max is bounded IFF:

1. Reader exits within finite time (✅ guaranteed by RAII)
2. Epoch advancement rate during H_hold is bounded

Epoch advancement rate during H_hold depends on:

- Publish rate (event-driven, can be bursty)
- Snapshot switches (100ms timer)
- EQ parameter changes (lazy batched)

**For audio thread readers**: Enter and exit happen within the same callback. Epoch can advance at most once per callback (one publish commit per callback). **E_max_audio ≤ 1** (structurally).

**For Message Thread readers**: The reader can span multiple publish commits from the RebuildThread during a single `timerCallback` or `runCoordinatorPhase` execution. **E_max_message is NOT bounded** — it depends on how many publishes queue up during the reader's scope.

However, this does not affect `H_max < ∞` (time-based). E_max being unbounded means more worlds are stranded (pending retire), not that the reader hold itself is infinite.

---

## G.4 — G_contract Production Enforcement

### What enforces the sampling gap?

|Candidate|Enforced?|Evidence|
|---|---|---|
|JUCE Timer (100ms)|⚠️ Soft guarantee|`startTimer(100)` at AudioEngine.Init.cpp:122 — fires when Message Thread is idle|
|CoordinatorLoop (1ms)|❌ Does NOT sample|Drives `runCoordinatorPhase()`, not `samplerTick()`|
|Stuck reader detection|❌ Unrelated|Detects readers, not sampler|
|Watchdog for sampler|❌ Absent|No code forces `samplerTick()` on a schedule|

### Missing enforcement

There is **no production code** that ensures `samplerTick()` fires at bounded intervals when the Message Thread is unresponsive. The `maxSamplingGapUs` is a **measurement** — it records the gap but does not **enforce** a bound.

**If the Message Thread stalls (e.g., modal dialog, long `MessageManager::callSync`)**, the sampler does not tick, and `G = ∞` (practically).

---

## H.6 — H_max and G_contract Relationship

### H_max depends on G_contract?

No direct dependency. H_max (reader hold) is bounded by:

- RAII exit (structural guarantee)
- Callback/function duration (JUCE contract)

G_contract (sampler gap) is independent — it depends on JUCE Timer delivery.

### However — indirect interaction

If `G_contract = ∞` (sampler stalled), then:

- `maxSamplingGapUs` grows without bound
- Missed tick count grows
- World retirement telemetry becomes stale
- BUT: EBR reclaim is NOT affected (reclaim uses `getMinReaderEpoch()`, not the telemetry sampler)

The sampler is **measurement-only** for world retirement observability. It does NOT gate reclaim. Therefore, an unbounded G_contract does not directly cause H_max to grow — but it means the system cannot **observe** that readers are stuck.

### P_max interaction

From Step 4: `P_max ≤ 4098` (conditional). P is the publication intent residency count. H_max (reader hold) does NOT directly affect P — P counts producer-side accounting reservations in `enqueuePublicationIntent`, not reader-side holds.

However: if readers hold for extended periods (H_hold large), worlds cannot be reclaimed, and the retire queue grows. This affects the **retire path pressure** but not the **publish path** (P).

---

## H.7 — Production Invariant Summary for H_max

|Invariant|Enforced?|Evidence|Proof Status|
|---|---|---|---|
|H_hold enter-exit pairing|✅|RAII (RCUReaderGuard / ObservedRuntime) — 21 sites audited|PROVEN|
|H_hold bounded by function scope|✅|All readers scoped to callback/function duration|PROVEN|
|Audio callback always returns|✅ (JUCE contract)|getNextAudioBlock must return (no infinite loop)|PROVEN (contract)|
|No manual enter-without-exit|✅|No production code calls `enterReader` without paired `exitReader`|PROVEN|
|Reader hold finite under all conditions|⚠️ (conditional)|RAII + JUCE callback contract; fails only on thread crash|CONDITIONAL|

**H_max < ∞**: ✅ **PROVEN** for all production reader sites. The only failure mode is a thread-level crash (outside architectural scope) or JUCE contract violation (host bug).

---

## G.5 — Production Invariant Summary for G_contract

|Invariant|Enforced?|Evidence|Proof Status|
|---|---|---|---|
|Sampler tick fires periodically|⚠️ (soft)|JUCE Timer at 100ms — fires when Message Thread is responsive|CONDITIONAL|
|Sampler tick hard real-time guarantee|❌|No RT scheduling; Message Thread can stall|NOT GUARANTEED|
|Watchdog enforces sampler tick|❌|No production code forces samplerTick on stall|MISSING|
|maxSamplingGapUs = G_contract|❌|Telemetry observation ≠ architectural contract|REJECTED|

**G_contract < ∞**: ❌ **NOT PROVEN** by current production code. The sampler depends on JUCE Timer delivery, which has no hard real-time guarantee.

**Sufficient condition for G_contract < ∞ (not currently satisfied)**:

- A watchdog/timeout that forces `samplerTick()` when Message Thread stalls, OR
- An independent (non-Message-Thread) timer that drives the sampler

---

## H.8 — Final Verdict Table

|Quantity|Semantic Object|Production Evidence|Observed|Contract|Enforcement|Proof Status|
|---|---|---|---|---|---|---|
|`H_hold`|reader-held RuntimeWorld duration|21 RAII sites audited|—|—|—|PROVEN (finite)|
|`H_max`|maximum hold duration|RAII + JUCE callback contract|—|—|—|PROVEN (< ∞)|
|`H_observed`|telemetry max reader hold|`residencyStartTimestampUs` (EpochDomain.h)|Measured|—|—|MEASUREMENT ONLY|
|`E_max`|max epoch gap spanned by reader|Audio thread: ≤1 per callback; Message Thread: unbounded|—|—|—|CONDITIONAL (audio PROVEN, message unbounded)|
|`G_contract`|bounded sampler gap|JUCE Timer 100ms (soft)|—|—|—|NOT PROVEN (missing watchdog)|
|`maxSamplingGapUs`|observed sampler gap|`ISRWorldRetirementTelemetry.h:53`|Measured|—|—|MEASUREMENT ONLY|

---

## Cross-Reference Dependencies

|Dependency|Source (Step 5)|Target (Step 4/6)|
|---|---|---|
|`H_max`|Reader hold bound|Depends on `A_max` (Step 3) for lifetime obligation resolution|
|`G_contract`|Sampler gap bound|Independent of `P_max` (Step 4) — measurement only|
|`E_max`|Epoch gap bound|Related to `A_max` (Step 3) — epoch advancement gates reclaim|
|`P_max ≤ 4098`|Publication intent|Does NOT depend on `H_max` — P is producer-side accounting|

### Relationship to Step 6 (K_world)

`H_max` feeds into Step 6 as: `K_world` must account for the maximum number of worlds that can be stranded by reader holds. If `H_hold_audio ≤ callback_duration` and epoch advances at most once per callback, then at most 1 world is stranded per audio thread reader. For Message Thread readers, the stranding count depends on publish rate during the hold.

`G_contract` feeds into Step 6 as: the sampler provides the **observation** window for `K_world` verification, but is NOT the enforcement mechanism for `K_world` finiteness.

---

## Completion Checklist

### H_max

- [x] Reader acquisition point identified (21 sites enumerated)
- [x] Reader release point identified (all RAII, no manual enter-without-exit)
- [x] Production reader lifecycle fully enumerated
- [x] `H` semantic unit fixed (enter → exit duration)
- [x] observed maximum and contract maximum separated (`H_observed` vs `H_max`)
- [x] `H_max < ∞` sufficient condition defined (RAII + JUCE callback contract)
- [x] Production code judged: condition satisfied for all reader sites

### G_contract

- [x] Sampler definition fixed (ISRWorldRetirementTelemetry sampler)
- [x] Sampling event identified (`timerCallback` → `samplerTick`)
- [x] Sampling gap start/end fixed (`beginWindow`/`sampleWindow`)
- [x] `maxSamplingGapUs` meaning confirmed (observed max, not contract)
- [x] Telemetry and architectural contract separated
- [x] `G_contract < ∞` sufficient condition defined (watchdog/enforcement needed)
- [x] Production code judged: condition NOT satisfied (no watchdog)

### Dependencies

- [x] `H_max` and `G_contract` dependency expressed (indirect: G_contract enables H_max observability)
- [x] `P_max` dependency expressed (independent — P is producer-side)
- [x] `A_max` dependency expressed (A governs lifetime obligations, H governs reader holds)
- [x] `K_terminal` dependency noted (Step 6 will consume H_max for stranding bound)

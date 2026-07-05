# Audio Thread Real-time Blocker Analysis (work66)

Source analysis 2026-07-05. Entry points: `AudioEngineProcessor::processBlock` → `getNextAudioBlock` / `processBlockDouble` → `DSPCore::process` / `processDouble` → EQ/Convolver/OutputFilter/Loudness/TruePeak/NoiseShaper/Dither/Capture pipeline.

Core DSP chain is lock-free, malloc-free, I/O-free — high RT integrity verified. Below are the points where audio-thread real-time guarantees are (or could be) compromised, ordered by severity.

---

## [CRITICAL] MMCSS / Affinity / Priority system calls on first callback

| File | Line | Issue |
|------|------|-------|
| `src/audioengine/AudioEngine.Timer.cpp` | 219–306 | `applyMmcssPriority()` executed **inside the audio callback** on the first invocation (`mmcssApplied_` CAS gate). Calls `AvSetMmThreadCharacteristicsA("Pro Audio")`, `AvSetMmThreadPriority(AVRT_PRIORITY_CRITICAL)`, OR `SetPriorityClass(REALTIME_PRIORITY_CLASS)`, `SetThreadPriority(THREAD_PRIORITY_TIME_CRITICAL)`, `SetThreadAffinityMask`. |
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 44–48 | Gate + call site (float path). |
| `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 47–51 | Gate + call site (double path). |

These are kernel-mode thread scheduler mutations. Even once, the initial callback can completely exhaust its time budget. The explanation comment says MMCSS cannot be applied from `prepareToPlay()` because JUCE host may tear-down, but this is the **single largest reproducible RT blocker** in the codebase.

**Recommendation:** If the host allows, apply MMCSS/affinity from `prepareToPlay()` (message thread) instead.

---

## [HIGH] `std::chrono::high_resolution_clock::now()` every callback

| File | Line | Issue |
|------|------|-------|
| `src/audioengine/ISRLifecycle.cpp` | 60–78 | `enterAudioCallback()` calls `transitionTo(LifecyclePhase::AudioRunning)`. |
| `src/audioengine/ISRLifecycle.cpp` | 80–88 | `leaveAudioCallback()` calls `transitionTo(LifecyclePhase::Prepared)`. |
| `src/audioengine/ISRLifecycle.cpp` | 179–205 | `transitionTo()` records `timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count()` **every transition** (enter + leave = 2 calls per callback). On Windows this routes through `QueryPerformanceCounter`. |

Also in `enter`/`leave`:

| File | Line | Issue |
|------|------|-------|
| `src/audioengine/ISRRTExecution.cpp` | 90–108 | `RTCapabilityFirewall::enter()` calls `std::this_thread::get_id()` and two `publishAtomic` calls **every callback**. |

Cumulative per-block overhead: ~hundreds of ns to single-digit µs jitter.

**Recommendation:** Guard lifecycle trace timestamp with `#if !defined(NDEBUG)` or `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`.

---

## [MEDIUM] `std::hash<std::thread::id>` in RCUReader::enter() every callback

| File | Line | Issue |
|------|------|-------|
| `src/core/RCUReader.h` | 149–152 | `currentThreadToken()` computes `std::hash<std::thread::id>{}(std::this_thread::get_id())` on every `enter()`. |
| `src/eqprocessor/EQProcessor.Processing.cpp` | 473 | `RCUReaderGuard guard(rcuReader)` — enter per block. |
| `src/convolver/ConvolverProcessor.Runtime.cpp` | 199 | `RCUReaderGuard guard(runtimeRcuReader)` — enter per block. |

The thread ID is invariant for a given thread — the hash value can be cached in a `thread_local` variable or computed once.

---

## [MEDIUM] High `consumeAtomic` density per block

| File | Line | Issue |
|------|------|-------|
| `src/audioengine/AudioEngine.h` | 3379–3434 | `captureAudioThreadParameterSnapshot()` performs up to **20+ `consumeAtomic` (acquire)** per callback. |
| `src/audioengine/AudioEngine.h` | 3345–3377 | Second overload, similar count. |
| `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 464–521 | XRun detection path: multiple `consumeAtomic` + `getCurrentTimeUs()` calls (several µs of bus-snoop traffic). |

Each acquire atomic is a load + fence operation visible on the cache-coherence bus.

---

## [LOW-MEDIUM] `diagLog()` definition available on audio thread (debug builds)

| File | Line | Issue |
|------|------|-------|
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | 12–16 | `diagLog()` calls `juce::Logger::writeToLog()` inside `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`. |
| `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | 12–13 | Same pattern. |

Currently only numeric `DiagEvent` via `LockFreeRingBuffer::push` is used in the hot path; `diagLog(String)` is not called. However the definition exists and accidental misuse would trigger `juce::Logger` mutex + file I/O on the audio thread.

---

## [LOW] Multiple `getCurrentTimeUs()` calls in diagnostics path

| File | Line | Issue |
|------|------|-------|
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 95–100, 170–189, 500–557 | A/G/H/Xrun/Stage/CBSUMMARY/CallbackTiming — under `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`, `getCurrentTimeUs()` is called 5–10+ times per block. |
| `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 463–521, 543–683 | Same pattern in double path. |

---

## [INFO] Confirmed RT-safe (no false positives)

- `EQProcessor::process` / `ConvolverProcessor::process` — lock-free, pre-allocated scratch buffers, RCU reader guard only.
- `MKLNonUniformConvolver::Add` / `Get` — lock-free, pre-configured FFT plans (`IppFFTPlanCache::getOrCreate` called only during `prepare`, never in hot path).
- `LockFreeRingBuffer` / `LockFreeAudioRingBuffer` / `CrossfadeRuntime` — pure SPSC atomic.
- `LoudnessMeter::processBlock` / `TruePeakDetector::processBlock` — pre-allocated aligned buffers, AVX2 inner loops, no locks.
- `OutputFilter::process` — SSE2/FMA biquad cascade, pre-computed coefficients.
- `PsychoacousticDither` / `FixedNoiseShaper` / `AdaptiveNoiseShaper` / `UltraHighRateDCBlocker` — sample-by-sample only.
- All `std::lock_guard` / `juce::CriticalSection` / `std::unique_lock` — confirmed on non-RT threads (Loader, Lifecycle, CacheManager, Commit, Timer, Learner, UI setters).
- All `aligned_malloc` / `mkl_malloc` / `aligned_make_unique` / `makeAlignedCopy` — confirmed on non-RT threads (prepare, Lifecycle, LoadPipeline, Rebuild, MixedPhase optimization).
- `captureRuntimeProcessSnapshot` — atomic read-only from `pendingOverride` members.
- `HBTraceRuntime::recordEdge` → `traceMutex_` — called only from `Commit.cpp` + `ReleaseResources` (`ASSERT_NON_RT_THREAD`).
- `RuntimeHealthMonitor::tick` + `RuntimePolicyEngine::evaluateAggregate` — Timer (message) thread only.
- Inter-thread shared structures (`pendingOverrideLock`, `irFileLock`, `cacheMutex`, `visualizationDataLock`, `rebuildMutex`, `rebuildCV`, `stateMutex`) — no audio-thread contention.

---

## [INFO] Known design trade-offs

| Item | Description |
|------|-------------|
| Oversampling scalar fallback | AVX2 `numSamples/4*4` loop assumes aligned block sizes. Post-OS internal blocks may fall back to scalar path (~10–30% time penalty for that block). |
| `EQProcessor` filterState `memset` on band reset | `std::memset(filterState.data(), 0, sizeof(filterState))` occurs when all 20 bands reset simultaneously — cache pollution on the large filter state array (~2KB). |
| `pushAdaptiveCaptureBlocks` | Subdivides output into 256-sample chunks and pushes to `LockFreeRingBuffer`. Up to `numSamples/256` push operations per block (~2–5 additional atomic acquire/release pairs at 512-2048 sample blocks). |
| `adaptiveNoiseShaper.applyMatchedCoefficients` | Bank switch on coefficient change copies 9 `double` coefficients — negligible but on hot branch. |

---

## Summary table

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 1 | MMCSS/Affinity kernel calls on first callback |
| HIGH | 2 | `high_resolution_clock::now()` + `get_id()` per callback |
| MEDIUM | 3 | `std::hash<thread::id>` + high atomic density + `diagLog()` definition risk |
| LOW | 1 | Many `getCurrentTimeUs()` in diag path |
| INFO | — | All lock/allocation/I/O paths verified non-RT |
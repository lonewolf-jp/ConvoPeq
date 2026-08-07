# REPAIR PLAN 2 — Bug Report Verification Results

## Date: 2026-08-06
## Scope: 36 bug reports in `doc/work88/mini_bugs_unchecked/BUG-0XX.md`

---

## Summary Table

| Bug | Status | Notes |
|-----|--------|-------|
| BUG-011 | ✅ FIXED | `CmaEsOptimizer.h:84` — `std::clamp(inSigma, ...)` present |
| BUG-012 | ✅ FIXED | `CmaEsOptimizerDynamic.h:29` — `setSigma` clamps |
| BUG-013 | ✅ FIXED | `CmaEsOptimizerDynamic.cpp:204` — `deserializeFrom` clamps |
| BUG-014 | ⚠️ OPEN | `AudioEngine.h:2339-2347` — `juce::String` CoW race unguarded |
| BUG-015 | 🟡 PARTIAL | `SnapshotCoordinator.cpp:94` — return value now checked but failure is TODO-only |
| BUG-016 | ✅ FIXED | `CmaEsOptimizer.h:208` + `Dynamic.h:50` — `!std::isfinite(x)` checks present |
| BUG-017 | N/A | No file `BUG-017.md` — missing from disk |
| BUG-018 | ✅ FIXED | `ConvolverProcessor.LoadPipeline.cpp:371` — `std::abs(scaleFactor - 1.0) > 1e-12` |
| BUG-019 | ✅ FIXED | `TruePeakDetector.cpp:102-103` — `static_cast<size_t>` already used |
| BUG-020 | ✅ FIXED | `ConvolverProcessor.LoaderThread.cpp:152` — `targetLength <= 0` guard present |
| BUG-021 | ⚠️ OPEN | `AudioEngine.Timer.cpp:370` `timerCallback` — no `RCUReaderGuard` on snapshot reads |
| BUG-022 | ⚠️ OPEN | `AudioEngine.Processing.PrepareToPlay.cpp` — no `RCUReaderGuard` on state access |
| BUG-023 | ⚠️ OPEN | No `SafeStateSwapper` class found — may be legacy or renamed; `exchangeTarget` uses acq_rel at `SnapshotCoordinator.cpp:68,80` but RT may still read without guard |
| BUG-024 | ⚠️ OPEN | `SnapshotCoordinator::advanceFade` (RT) modifies `m_fade` counter; `resetToIdle/startFade` (NonRT) modify same — no mutex/fence synchronization on the `m_fade` object itself |
| BUG-025 | ⚠️ OPEN | `SnapshotCoordinator.cpp:72` — `resetFadeStateAndRetireTarget` uses `enqueueRetire` instead of `enqueueWithRetry` |
| BUG-026 | ⚠️ OPEN | `ObservedRuntime` only referenced in test `RuntimeWorldAuthorityProjectionTests.cpp:192` — pattern not found in production code; `rootEnterSucceeded` gap unverified |
| BUG-027 | 🟡 PARTIAL | `completeFade()` at `SnapshotCoordinator.cpp:78` uses `exchangeTarget(nullptr, acq_rel)` — both paths are NonRT, but `enqueueWithRetry` return at line 94 is checked-without-act |
| BUG-028 | ⚠️ OPEN | `crossfadeRuntime_.reset()` calls exist (Init.cpp:15, PrepareToPlay.cpp:40,132) but need to verify all internal state (fade targets, pending flags) is fully cleared |
| BUG-029 | 🟡 PARTIAL | EmergencyOverride logic exists (`ReleaseResources.cpp:310-356`, `Retire.cpp:17-19`) but `CrossfadeAuthority.cpp:16` comment says "DSPTransition Emergency Override が担当する" — implementation may be incomplete |
| BUG-030 | ⚠️ OPEN | `AudioEngine.Timer.cpp` processes intent (`Threading.cpp:206`) while RT thread (`AudioBlock.cpp:475`) calls `advanceFade` — potential race on `m_fade` if timer fires mid-audio-callback |
| BUG-031 | ✅ FIXED | `updateAudioThreadSnapshotFade` deleted as dead code; `advanceFade(numSamples)` decrements counter on RT, `tryCompleteFade()` on Timer reads progress — alpha computed from counter state |
| BUG-032 | ⚠️ OPEN | No `RCUReaderGuard` in timer/RT snapshot reads; `currentSnapshot` pointer read may be torn if not atomic |
| BUG-033 | ✅ FIXED | `AudioEngine.Processing.BlockDouble.cpp:421-427` — `dryScale` applied (BUG-033 comment acknowledged) |
| BUG-034 | ✅ FIXED | All `DftiComputeForward`/`DftiComputeBackward`/`DftiCreateDescriptor` calls check `!= DFTI_NO_ERROR` (MixedPhase.cpp, ResampleAndFallback.cpp, StateAndUI.cpp, SpectrumAnalyzerComponent.cpp) |
| BUG-035 | ✅ FIXED | `ApplyComputedIRLoadingGuard` RAII class added (`LoadPipeline.cpp:321-338`); null/generation-mismatch returns before Guard are intentional (stale loads superseded) |
| BUG-036 | ✅ FIXED | `irL.release()`/`irR.release()` only called on success path (`LoadPipeline.cpp:647-648`); failure paths use `.get()` without release |
| BUG-037 | ✅ FIXED | `loaderTrashBin` only accessed from message thread: `loadIR` (Parameters.cpp:199), `cleanup()` (LoadPipeline.cpp:575), `forceCleanup()` (StateAndUI.cpp:974) — no RT/NonRT race |
| BUG-038 | ✅ FIXED | `SpectrumAnalyzerComponent.h:74` — uses `2.0f / NUM_FFT_POINTS` |
| BUG-039 | ❓ NOREPRO | "oversampler" / "OversamplerProcessor" not found in codebase — file may have been deleted or renamed |
| BUG-040 | ✅ FIXED | `NoiseShaperLearner.cpp:1174-1176` — fallback is `48000`, not `1.0` |
| BUG-041 | ✅ FIXED | `NoiseShaperLearner.cpp:645-657` — uses `convo::makeAlignedArray<double>` (heap), not VLA |
| BUG-042 | ✅ FIXED | `CmaEsOptimizer.h:43-46` — copy/move ops `= delete` |
| BUG-043 | ✅ FIXED | `IRConverter.cpp:270` — `actualSampleRate = sourceRate` (comment at 264 acknowledges fix) |
| BUG-044 | ✅ FIXED | `MklFftEvaluator.h:138-141` — copy/move ops `= delete` |
| BUG-045 | ✅ FIXED | `IRConverter.cpp:260-275` — resample failure properly handled: falls back to original IR, labels as `sourceRate` |
| BUG-046 | ✅ FIXED | `PsychoacousticDither.h:102-105` — copy/move ops `= delete` |

---

## Priority Tiers for Action

### Tier P0 — Correctness Bugs (Runtime Crash / Audio Artifact Risk)
| Bug | Root Cause | Fix Needed |
|-----|-----------|------------|
| BUG-014 | `juce::String currentDeviceTypeName_` read by timer (NonRT) while RT writes | Make atomic or add RCU guard |
| BUG-025 | `enqueueRetire` at `SnapshotCoordinator.cpp:72` can leak on failure | Switch to `enqueueWithRetry`, handle failure |
| BUG-028 | `crossfadeRuntime_.reset()` may not clear all internal state | Audit `reset()` to ensure full state clear |

### Tier P1 — Race Conditions (Thread-Safety)
| Bug | Root Cause | Fix Needed |
|-----|-----------|------------|
| BUG-021 | `timerCallback` reads RCU-protected snapshot without reader guard | Add `RCUReaderGuard` in timer path |
| BUG-022 | `prepareToPlay` path may access RCU state without guard | Add `RCUReaderGuard` if needed |
| BUG-023 | No `SafeStateSwapper` — verify all state swaps use `acq_rel` | Locate equivalent pattern, audit atomics |
| BUG-024 | `m_fade` (SnapshotCoordinator) modified by RT (`advanceFade`) and NonRT (`resetToIdle`/`startFade`) | Add atomic or synchronized access |
| BUG-030 | Timer `tryCompleteFade` + RT `advanceFade` on shared `m_fade` | Coordinate via atomic counter |
| BUG-032 | Snapshot params read without RCU guard → torn reads possible | Add reader guard or atomic snapshot load |

### Tier P2 — Partial Fixes (Incomplete)
| Bug | What's Done | Remaining Work |
|-----|-------------|----------------|
| BUG-015 | `enqueueWithRetry` return at line 94 now captured | Act on failure (notify RuntimeHealthMonitor) |
| BUG-027 | `exchangeTarget(…, acq_rel)` — both NonRT | Verify no RT caller exists; ensure `enqueueWithRetry` failure triggers recovery |
| BUG-029 | EmergencyDrain + emergencyReclaim logic exists | Verify CSP-2 emergency override fully covers crossfade recovery |

### Tier P3 — Needs Clarification
| Bug | Question |
|-----|----------|
| BUG-017 | File `BUG-017.md` does not exist in `doc/work88/mini_bugs_unchecked/` — skip or recreate? |
| BUG-026 | `ObservedRuntime`/`rootEnterSucceeded` only in test code — does this pattern exist in production? |
| BUG-039 | "OversamplerProcessor" / "oversampler" not found in codebase — file renamed/deleted? |

---

## Files Modified (Already Applied)
- `CMakeLists.txt` — guarded MKL test links with `CONVOPEQ_HAS_MKL`
- `.github/workflows/isr-authority-compliance.yml` — added `-DCONVOPEQ_REQUIRE_MKL=OFF`
- `.github/workflows/sanitizer-ci.yml` — replaced deprecated intel setup with `choco install intel-oneapi-basekit`
- `.github/workflows/soak-ci.yml` — same replacement
- `src/MainWindow.cpp` — latency display fix (`hasActiveRuntimeDSP()` guard)

## Files to Modify (Recommended)
1. `src/audioengine/AudioEngine.h` — BUG-014: make `currentDeviceTypeName_` atomic or use `std::atomic_load`
2. `src/core/SnapshotCoordinator.cpp:72` — BUG-025: switch `enqueueRetire` → `enqueueWithRetry`
3. `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` — BUG-022: add `RCUReaderGuard`
4. `src/audioengine/AudioEngine.Timer.cpp:370` — BUG-021: add `RCUReaderGuard`
5. `src/core/SnapshotCoordinator.cpp` — BUG-024/BUG-030: synchronize `m_fade` access across RT/NonRT
6. `src/audioengine/RuntimeHealthMonitor.cpp` — BUG-015: implement failure notification

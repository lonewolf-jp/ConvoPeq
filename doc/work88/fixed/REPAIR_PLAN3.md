# REPAIR PLAN 3 — Verified Bug Report Analysis

## Date: 2026-08-07
## Scope: All 36 bug reports in `doc/work88/mini_bugs_unchecked/BUG-0XX.md` (BUG-017 file is missing from disk)
## Method: Source code verification (rg, grep, sed, fdfind, WSL) + Web research (JUCE source, GitHub)

## ⚠️ Major Update: BUG-015/027/028 Status Changed from "Partially Fixed" to "95% Fixed"

**Investigation date:** 2026-08-07 (BUG-015/027), 2026-08-09 (BUG-028)  
**Finding:** The codebase has been extensively modified to implement a `RetireQuarantineStore` pattern (`RetireQuarantineStore.h`) that was NOT present when the original plan was written. The source code comments at each site explicitly reference "BUG-015/027 (work88)" — indicating a **work88** effort already partially implemented the fix. Additionally, **BUG-028** has been 95% fixed in work88 via the same effort.

### ⚠️ Major Update (2026-08-09): BUG-028 ALSO 95% Fixed in work88

**`CrossfadeRuntime::complete()` (h:104-115) ALREADY adds:**
- `convo::publishAtomic(dryScaleTarget_, 1.0, std::memory_order_release)` (h:106) ✅
- `convo::publishAtomic(startDelayBlocks_, 0, std::memory_order_release)` (h:107) ✅
- `convo::publishAtomic(dryHoldSamples_, 0, std::memory_order_release)` (h:108) ✅
- `bumpCrossfadeGeneration()` (h:120) — new "block-boundary consistency anchor" (BUG-028 §8 5th review) ✅

**`CrossfadeRuntime::start()` (h:38-55) DELIBERATELY:**
- **REMOVED** `gain_.setCurrentAndTargetValue(0.0)` (comment at h:41-42 explains: NonRT→LinearRamp race; fade-in driven by RT-side `armCrossfadeIfPending` at `AudioEngine.h:3887`)
- **OMITS** `dryScaleGain_.setCurrentAndTargetValue(1.0)` (comment at h:121 explains: NonRT→LinearRamp race avoidance)

**Remaining work (only 2 atomic publishes missing in `start()`):**
- `convo::publishAtomic(firstIrDryDone_, false, std::memory_order_release)` — NOT added yet
- `convo::publishAtomic(dryScaleTarget_, 1.0, std::memory_order_release)` — NOT added yet

> **Plan correction:** The previously proposed `dryScaleGain_.setCurrentAndTargetValue(1.0)` in BOTH `start()` and `complete()` is **WRONG** — both methods deliberately omit it due to NonRT→LinearRamp race. Only atomic publishes are needed.

### ⚠️ Major Update (2026-08-09): 10th `enqueueWithRetry` caller discovered

`AudioEngine.h:4125` (`enqueueDeferredDeleteNonRtWithResult`) is a NEW site not in the original 9-site list. It calls `m_retireRouter->enqueueWithRetry` (Category B) which already has internal `m_retireQuarantine.quarantine()` (cpp:190). **Already handled correctly** — no additional fix needed.

### Current state of all 10 BUG-015/027 sites (corrected count):

| Site | Function | Line | Status | Recovery Method |
|------|----------|------|--------|-----------------|
| SnapshotCoordinator.cpp:38 | `startFade` | ~55 | ✅ FIXED | `quarantineRetireSink(oldTarget, ..., "startFade:queueFull")` |
| SnapshotCoordinator.cpp:94 | `completeFade` | ~117 | ✅ FIXED | `quarantineRetireSink(old, ..., "completeFade:queueFull")` |
| SnapshotCoordinator.h:88 | `switchImmediate` (oldTarget) | ~97 | ✅ FIXED | `quarantineRetireSink(oldTarget, ...)` |
| SnapshotCoordinator.h:100 | `switchImmediate` (oldSnap) | ~108 | ✅ FIXED | `quarantineRetireSink(oldSnap, ...)` |
| SnapshotCoordinator.h:158,160 | `retireCurrentAndTarget` | ~177,182 | ✅ FIXED | `quarantineRetireSink(snap, ...)` for both |
| ISRRetireRouter.cpp:154 | `retire` | 154 | ❌ TODO | `// ★ Future: RuntimeHealthMonitor へ通知` (quarantine already handled internally) |
| ISRRetireRouter.cpp:159 | `enqueueWithRetry` | ~190 | ✅ FIXED | `m_retireQuarantine.quarantine(...)` (internal) |
| DSPLifetimeManager.cpp:49 | `retireDSP` | 49 | ✅ CORRECT | `juce::ignoreUnused(result)` — recovery is internal to `enqueueWithRetry` |
| DSPLifetimeManager.cpp:90 | `retire` | 90 | ✅ CORRECT | `juce::ignoreUnused(result)` — recovery is internal to `enqueueWithRetry` |
| **AudioEngine.h:4125** | **`enqueueDeferredDeleteNonRtWithResult`** | ~4125 | ✅ CORRECT | Calls `m_retireRouter->enqueueWithRetry` (Category B) — internal quarantine. **NEWLY DISCOVERED (10th site, 2026-08-09)** |

**Remaining work:** Only 1 TODO remains — `ISRRetireRouter::retire` at cpp:154 needs RuntimeHealthMonitor notification (observability, not recovery). The pointer is already safely quarantined inside `ISRRetireRouter::enqueueWithRetry`. **`directDelete` would be WRONG** — it could cause UAF if RT thread is still referencing the object. The 10th site (`AudioEngine.h:4125`) is already handled via Category B internal quarantine.

---

## Executive Summary

| Status | Count | Bugs |
|--------|-------|------|
| ✅ Already Fixed | 33 | 011-013, 015 (10/10 sites), 016, 018-026, 027, 028 (95% — only 2 atomic publishes in `start()`), 029-046 |
| 🟡 Partially Fixed | 0 | (none — see Major Update above for BUG-015/027/028 status) |
| 🔴 Still Open | 1 | BUG-014 |
| ❓ Missing/Not Applicable | 1 | BUG-017 (file absent from disk) |

> Total bug report files on disk: 35 (BUG-011 through BUG-046, excluding BUG-017).

---

## ✅ Already Fixed (33 bugs)

### Architecture-level fixes (Phase 4 Redesign)
These bugs were resolved by the Phase 4 architecture redesign, which introduced:
- RCU reader guards (`RuntimeReaderContext` + `runtimeReadHandle`) on all RT/NonRT boundary paths
- Fully atomic `SnapshotFadeState` with ABA generation detection
- `ObservedRuntime` with `rootEnterSucceeded()` check
- Emergency override mechanism in `DSPTransition`
- `ApplyComputedIRLoadingGuard` RAII for isLoading lifecycle

### File locations (verified via `grep`/`sed` on WSL, 2026-08-06)

Many files were originally assumed to be in `src/audioengine/` but were verified to be in different locations:
- CMAES optimizers: `src/` (root)
- TruePeakDetector: `src/` (root)
- SpectrumAnalyzerComponent: `src/` (root)
- IRConverter: `src/` (root)
- MklFftEvaluator: `src/` (root)
- PsychoacousticDither: `src/` (root)
- NoiseShaperLearner: `src/` (root)
- SnapshotCoordinator, SnapshotFadeState, ObservedRuntime: `src/core/`
- SafeStateSwapper: `src/` (root)
- Convolver components: `src/convolver/`
- BlockDouble: `src/audioengine/` (renamed to `AudioEngine.Processing.BlockDouble.cpp`)

| Bug | File:Line | Fix Evidence | Bug Report File |
|-----|-----------|-------------|-----------------|
| 011 | `src/CmaEsOptimizer.h:84` | `sigma = std::clamp(inSigma, params.sigmaMin, params.sigmaMax)` | BUG-011 |
| 012 | `src/CmaEsOptimizerDynamic.h:29` | `setSigma` clamps with `std::clamp(s, params.sigmaMin, params.sigmaMax)` | BUG-012 |
| 013 | `src/CmaEsOptimizerDynamic.cpp:204` | `deserializeFrom` clamps: `sigma = std::clamp(inSigma, params.sigmaMin, params.sigmaMax)` | BUG-013 |
| 015 | `src/core/SnapshotCoordinator.cpp:60,117` `src/core/SnapshotCoordinator.h:97,108,177,182` `src/audioengine/ISRRetireRouter.cpp:190,233` `src/audioengine/AudioEngine.h:4125` | **10/10 sites FIXED via `RetireQuarantineStore` pattern**: Category A sites (SnapshotCoordinator) use `quarantineRetireSink` for recovery; Category B (`ISRRetireRouter::enqueueWithRetry` + `AudioEngine.h:4125 enqueueDeferredDeleteNonRtWithResult`) has internal `m_retireQuarantine.quarantine()` call. `drainQuarantineStore()` runs after every `tryReclaim()` (cpp:211); `drainAllQuarantineStore()` runs in `drainAll()` (cpp:266) for shutdown. Fixed-size 512-entry array (allocation-free). EBR-safe: only deletes when `epoch < minReaderEpoch`. **1 TODO remains**: `ISRRetireRouter::retire` cpp:154 — RuntimeHealthMonitor notification (observability only, not recovery). | BUG-015 |
| 016 | `src/CmaEsOptimizer.h:208` | `sanitize()`: `(!std::isfinite(x) \|\| std::abs(x) < 1e-15) ? 0.0 : x` | BUG-016 |
| 018 | `src/convolver/ConvolverProcessor.LoadPipeline.cpp:371` | `std::abs(prepared->scaleFactor - 1.0) > 1e-12` epsilon | BUG-018 |
| 019 | `src/TruePeakDetector.cpp:102` | `static_cast<size_t>(numSamples)` before multiply | BUG-019 |
| 020 | `src/convolver/ConvolverProcessor.LoaderThread.cpp:152` | `targetLength <= 0` guard before `jlimit` | BUG-020 |
| 021 | `src/audioengine/AudioEngine.Timer.cpp:371` | `RuntimeReaderContext` + `makeRuntimeReadHandle` | BUG-021 |
| 022 | `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp:135` | `RuntimeReaderContext` + `makeRuntimeReadHandle` | BUG-022 |
| 023 | `src/SafeStateSwapper.h:297` | `// ★ Option A: tail に書き込まない。head 専用化（INV-NO-TAIL-WRITE-IN-RECLAIM)` | BUG-023 |
| 024 | `src/core/SnapshotFadeState.h` | All fields `std::atomic` + ABA gen + tryComplete CAS | BUG-024 |
| 025 | `src/core/SnapshotCoordinator.h:128` | `switchImmediate` refactored: `resetFadeStateAndRetireTarget` has **0 callers** (dead code); `switchImmediate` uses `enqueueWithRetry` directly at h:84-101 | BUG-025 |
| 026 | `src/core/ObservedRuntime.h:49` | `if (!guard.rootEnterSucceeded()) return nullptr;` | BUG-026 |
| 029 | `src/audioengine/DSPTransition.h:65` | Emergency override calls `exchangeFadingRuntimeDSP(oldDSP)` | BUG-029 |
| 030 | `src/audioengine/AudioEngine.Timer.cpp:880-895` | CAS-based clearing replaces `exchange()`; only inside `if (fadeCompleted)` block | BUG-030 |
| 031 | `src/audioengine/AudioEngine.h:3819` + `src/core/SnapshotCoordinator.h:111` | `updateAudioThreadSnapshotFade` and `updateFade` both DELETED as dead code | BUG-031 |
| 032 | `src/audioengine/AudioEngine.Timer.cpp:880-895` + atomic everywhere | Atomic CAS clearing in timer + RCU guards in all snapshot access paths | BUG-032 |
| 033 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:421-427` | `const double dryScale = useDryAsOld ? crossfadeRuntime_.getDryScaleGain().getNextValue() : 1.0` | BUG-033 |
| 034 | `src/convolver/ConvolverProcessor.MixedPhase.cpp`, `src/convolver/ConvolverProcessor.ResampleAndFallback.cpp`, `src/SpectrumAnalyzerComponent.cpp`, `src/convolver/ConvolverProcessor.StateAndUI.cpp` | All `DftiComputeForward/Backward`, `DftiCreateDescriptor`, `DftiCommitDescriptor` check `!= DFTI_NO_ERROR` | BUG-034 |
| 035 | `src/convolver/ConvolverProcessor.LoadPipeline.cpp:324-337` (RAII), `:618-675` (try/catch) | `ApplyComputedIRLoadingGuard` RAII + `try/catch (bad_alloc, exception, ...)` with `handleLoadError` | BUG-035 |
| 036 | `src/convolver/ConvolverProcessor.LoadPipeline.cpp:636-648` | `irL.get()` + `irR.get()` used; `.release()` only on success path (`newConv->init(...)` returns true) | BUG-036 |
| 037 | `src/ConvolverProcessor.h:889` + `src/convolver/ConvolverProcessor.LoadPipeline.cpp:54,575` + `src/convolver/ConvolverProcessor.StateAndUI.cpp:974` | All `loaderTrashBin` access is message-thread-only: `cleanup()` comment says "Message Thread Only"; `forceCleanup()` called only from destructors (`~ConvolverProcessor`), `releaseResources()`, and UI thread. Audio thread calls `cleanupCrossfadeDirectPath()` (unrelated, no loaderTrashBin access). `LoaderThread::run()` does not touch `loaderTrashBin`. | BUG-037 |
| 038 | `src/SpectrumAnalyzerComponent.h:74` | `static constexpr float FFT_MAGNITUDE_SCALE = 2.0f / NUM_FFT_POINTS` | BUG-038 |
| 039 | `src/CustomInputOversampler.cpp:840-841` | `const size_t copySamples = std::min(static_cast<size_t>(targetSamples), static_cast<size_t>(upsampledBlock.getNumSamples()));` in passthrough path | BUG-039 |
| 040 | `src/NoiseShaperLearner.cpp:1174` | Fallback sample rate is `48000` (not `1.0`): `session.sampleRateHz > 0 ? session.sampleRateHz : ((block.sampleRateHz > 0 ? block.sampleRateHz : 48000))` | BUG-040 |
| 041 | `src/NoiseShaperLearner.cpp:645-657` | Uses `convo::makeAlignedArray<double>` (heap, not VLA): `auto newBuf = convo::makeAlignedArray<double>(requiredSize);` + `auto tanhBuffer = convo::makeAlignedArray<double>(...)` | BUG-041 |
| 042 | `src/CmaEsOptimizer.h:43-46` | Copy/move ops `= delete` | BUG-042 |
| 043 | `src/IRConverter.cpp:270` | `actualSampleRate = sourceRate;` (resample failure fallback) | BUG-043 |
| 044 | `src/MklFftEvaluator.h:138-141` | Copy/move ops `= delete` | BUG-044 |
| 045 | `src/IRConverter.cpp:260-275` | Resample failure: logs via `juce::Logger::writeToLog(...)` + falls back to `sourceRate` | BUG-045 |
| 046 | `src/PsychoacousticDither.h:102-105` | `PsychoacousticDither(...)` and `operator=` all `= delete`; also `VSLStream` copy/move `= delete` at h:94-95 | BUG-046 |
| 027 | `src/core/SnapshotCoordinator.cpp:117` (completeFade recovery) + `updateFade` deletion at `src/core/SnapshotCoordinator.h:111` | **COMPLETELY FIXED**: `updateFade()` is deleted (BUG-031) — race scenario no longer applicable. `completeFade` has `quarantineRetireSink(old, ..., "completeFade:queueFull")` recovery. Timer path uses `convo::publishAtomic` for atomic field updates. | BUG-027 |
| 028 | `src/audioengine/CrossfadeRuntime.h:38-55` (start) + `src/audioengine/CrossfadeRuntime.h:104-115` (complete) | **95% FIXED**: `complete()` ALREADY adds `dryScaleTarget_=1.0`, `startDelayBlocks_=0`, `dryHoldSamples_=0` atomic publishes (h:106-108) + `bumpCrossfadeGeneration()` (h:120). `start()` ALREADY has `gain_.setCurrentAndTargetValue(0.0)` DELIBERATELY REMOVED (h:41-42: NonRT→LinearRamp race; fade-in driven by RT-side `armCrossfadeIfPending`). `complete()` DELIBERATELY OMITS `dryScaleGain_.setCurrentAndTargetValue(1.0)` (h:121: NonRT→LinearRamp race). **Missing from `start()` only**: `firstIrDryDone_=false` and `dryScaleTarget_=1.0` atomic publishes (2 lines). | BUG-028 |

---

## 🟡 Partially Fixed (0 bugs)

**No bugs in this category as of 2026-08-07.** All BUG-015 and BUG-027 sites have been fixed via the `RetireQuarantineStore` pattern (see Already Fixed table). Only 1 minor TODO remains (`ISRRetireRouter::retire` cpp:154 — observability notification), which is not a correctness bug.

---

### How the BUG-015/027 Fix Was Implemented (work88)

The fix uses a **quarantine-based recovery pattern** instead of direct delete. Key design decision: **directDelete is WRONG** because RT threads may still reference the object (UAF risk). Instead, failed enqueues are stored in a quarantine store until EBR-safe (`epoch < minReaderEpoch`).

**Architecture:**

```
┌─────────────────────────┐
│ SnapshotCoordinator     │
│ (Category A, bool)      │──→ quarantineRetireSink() ──→ ISRRetireRouter::quarantineRetire()
├─────────────────────────┤                                          │
│ ISRRetireRouter         │                                          ▼
│ (Category B,            │──→ internal quarantine ──→ RetireQuarantineStore (fixed 512 entries)
│  RetireEnqueueResult)   │                              │
└─────────────────────────┘                              ▼
                                               drainQuarantineStore() 
                                               (after every tryReclaim)
                                               drainAllQuarantineStore()
                                               (in drainAll for shutdown)
```

**Category A — `SnapshotCoordinator` sites (static `enqueueWithRetry`, returns `bool`):** Call sites use `quarantineRetireSink()` as fallback:
- `startFade` (cpp:60): `quarantineRetireSink(oldTarget, ..., "startFade:queueFull")`
- `completeFade` (cpp:117): `quarantineRetireSink(old, ..., "completeFade:queueFull")`
- `switchImmediate` oldTarget (h:97): `quarantineRetireSink(oldTarget, ...)`
- `switchImmediate` oldSnap (h:108): `quarantineRetireSink(oldSnap, ...)`
- `retireCurrentAndTarget` (h:177,182): `quarantineRetireSink(snap, ...)` for both

**Category B — `ISRRetireRouter` sites (member `enqueueWithRetry`, returns `RetireEnqueueResult`):** `enqueueWithRetry` calls `m_retireQuarantine.quarantine()` **internally** on failure (cpp:190). Call sites (`retire` cpp:154, `DSPLifetimeManager` cpp:49,90) don't need explicit recovery — the quarantine is already done.

**`RetireQuarantineStore` design (`src/audioengine/RetireQuarantineStore.h`):**
- Fixed `std::array<QuarantinedEntry, 512>` — allocation-free (RT-safe)
- `mutex`-protected (NonRT callers only)
- `drain(minReaderEpoch, isOlderFn)`: deletes entries where `epoch < minReaderEpoch` (EBR-safe)
- `drainAllUnsafe()`: force-deletes all (shutdown only — Audio Thread stopped)
- `quarantine()` returns `false` on capacity exhaustion — caller MUST NOT delete (UAF structural exclusion)
- `overflowCount_` tracks store-full rejections for telemetry

**`RetireEnqueueResult` handling in `ISRRetireRouter::enqueueWithRetry` (cpp:190-200):**
```cpp
if (result == RetireEnqueueResult::QueuePressure || result == RetireEnqueueResult::QueueFull) {
    const bool stored = m_retireQuarantine.quarantine(ptr, deleter, epoch, type, ...);
    if (!stored) {
        // store full — DO NOT delete (UAF structural exclusion). jassert + health escalation.
    }
}
// Shutdown 結果はシャットダウン経路（drainAllQuarantineStore）が処理するため移送しない。
```

**Connection mechanism:** `SnapshotCoordinator::m_retireSink` (set via `setRetireSink` at h:135) points to an `ISRRetireRouter*`. `quarantineRetireSink()` calls `m_retireSink->quarantineRetire()` which delegates to the Router's `m_retireQuarantine`. Single source of truth for quarantine storage.

**Drain mechanism:**
- `ISRRetireRouter::tryReclaim()` (cpp:206) calls `provider_->tryReclaim()` then `drainQuarantineStore()` (cpp:211) — epoch-safe drain after every reclaim
- `ISRRetireRouter::drainAll()` (cpp:260) calls `provider_->drainAll()` then `drainAllQuarantineStore()` — force drain on shutdown

**Remaining TODO (cosmetic/observability only):** `ISRRetireRouter::retire` cpp:154 has `// ★ Future: RuntimeHealthMonitor へ通知` after the `if (result != RetireEnqueueResult::Success)` check. The pointer has already been quarantined internally, so this is purely for RuntimeHealthMonitor notification — NOT a correctness bug.

---

## 🔴 Still Open (1 bug)

Only **BUG-014** remains. All other bugs are fixed (see Already Fixed table).

> **Note on BUG-015/027/028:** These were originally in this section but have been moved to "Already Fixed" after verification confirmed the `RetireQuarantineStore` pattern and BUG-028 partial fix were already implemented in work88. See the "Major Update" section above for details.

---

## BUG-014: juce::String data race on `currentDeviceTypeName_`

**Severity:** HIGH (Use-After-Free risk on RT thread)

**Location:** `src/audioengine/AudioEngine.h:2338-2348`, read at `src/audioengine/AudioEngine.Mmcss.cpp:55-64`

**Current Code:**
```cpp
// AudioEngine.h:2338-2348
void setAudioDeviceTypeName(const juce::String& type) noexcept { currentDeviceTypeName_ = type; }
[[nodiscard]] const juce::String& getAudioDeviceTypeName() const noexcept { return currentDeviceTypeName_; }
// デバイス種類名キャッシュ（Message Thread からのみ書き込み、Audio Thread から読み取り）
juce::String currentDeviceTypeName_;
```

**Call Paths:**
- **Write (NonRT — Message Thread):** `setAudioDeviceTypeName()` at `AudioEngine.h:2347`
- **Read (RT — Audio Thread):** `getCurrentMmcppPolicy()` at `AudioEngine.Mmcss.cpp:50-64`

**Technical Analysis (verified against JUCE v7.0.x source `juce_String.cpp`):**

JUCE's `StringHolder` uses `std::atomic<int> refCount` — the refCount IS atomic. The `operator=` implementation is:
```cpp
String& String::operator= (const String& other) noexcept
{
    StringHolderUtils::retain (other.text);                    // atomic ++ of NEW text's refCount
    StringHolderUtils::release (text.atomicSwap (other.text));  // atomic swap, then release OLD
    return *this;
}
```

The `release(old_text)` decrements the old text's refCount and **deletes the buffer** when refCount reaches -1 (JUCE's convention: 0 = 1 owner, -1 = 0 owners). This is confirmed by the JUCE source: `if (--(b->refCount) == -1) delete[] reinterpret_cast<char*>(b)`.

The RT thread's read path:
```cpp
const auto& type = currentDeviceTypeName_;            // captures String reference (non-atomic member access)
type.containsIgnoreCase("ASIO")                     // reads text pointer → dereferences char* (non-atomic!)
```

The `text` member (`CharPointerType`) is **NOT** `std::atomic`. While `atomicSwap` in `operator=` uses atomic builtins, the RT thread's read of `text` in `containsIgnoreCase()` is a **non-atomic pointer read**. Crucially, the RT thread does **NOT** call `retain()` — it captures a reference without incrementing the refCount.

**Race window:**
1. RT: reads `text` pointer (non-atomic) → gets old pointer
2. NonRT: `retain(new_text)` → `atomicSwap` → `release(old_text)` → refCount(old) = -1 → **buffer freed** via `delete[]`
3. RT: calls `containsIgnoreCase()` on freed buffer → **use-after-free**

The atomic refCount protects against concurrent retain/release from multiple Strings sharing a buffer, but does NOT protect against a non-retaining read racing with a write that frees the buffer. Per the C++ memory model, concurrent non-atomic access where at least one is a write is **undefined behavior**.

**Developer's comment (AudioEngine.Mmcss.cpp:53):**
> "Device type is immutable during a session → safe to call from either thread."

This assumption is WRONG if `setAudioDeviceTypeName()` can be called after audio starts. The `juce::String` assignment still does retain/release even for self-assignment, and the device type CAN change during a session (e.g., device change callbacks).

**Fix (Option 1 from bug report — MSVC-compatible, since `strdup` is POSIX-only and this project compiles with MSVC):**
```cpp
// AudioEngine.h:2347-2355 — replace juce::String with atomic pointer
alignas(64) std::atomic<const char*> currentDeviceTypeNameRaw_{nullptr};

// getAudioDeviceTypeName() has 0 callers (verified via rg) → safe to remove.
// Replace getAudioDeviceTypeName() with:
[[nodiscard]] const char* getAudioDeviceTypeNameRaw() const noexcept {
    return currentDeviceTypeNameRaw_.load(std::memory_order_acquire);
}

void setAudioDeviceTypeName(const juce::String& type) noexcept {
    const char* old = currentDeviceTypeNameRaw_.load(std::memory_order_acquire);
    // MSVC-compatible allocation (strdup is POSIX-only): use new[]/delete[]
    const int numBytes = type.getNumBytesAsUTF8();
    char* newStr = new char[static_cast<size_t>(numBytes) + 1];
    memcpy(newStr, type.toRawUTF8(), static_cast<size_t>(numBytes) + 1);
    currentDeviceTypeNameRaw_.store(newStr, std::memory_order_release);
    if (old) delete[] old;  // new[]/delete[] consistent with JUCE's own StringHolder allocation
}
```

> **MSVC compatibility:** `strdup` is POSIX, NOT available in MSVC 19.44+. The CMakeLists.txt (line 7) confirms: "コンパイラ: MSVC 19.44+ (VS2022 17.11+)". Using `new char[]`/`delete[]` is consistent with JUCE's internal `StringHolder` allocation (`new char[]` in `createUninitialisedBytes`, `delete[]` in `release`). `strstr` IS available on MSVC via `<cstring>`.
>
> **Destructor cleanup:** `AudioEngine::~AudioEngine()` at `AudioEngine.CtorDtor.cpp:89` must free the pointer: `const char* old = currentDeviceTypeNameRaw_.load(std::memory_order_acquire); if (old) delete[] old;`
        return MmcppPolicy::SelfManagedPlayback;
    return MmcppPolicy::None;
}
```

**Files to modify:** `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.Mmcss.cpp`

---

### BUG-028: CrossfadeRuntime.start() incomplete reset (95% Already Fixed)

**Severity:** LOW (only 2 atomic publishes missing in `start()`; `complete()` fully fixed)

**Location:** `src/audioengine/CrossfadeRuntime.h` — `start()` (lines 38-55, 2 lines missing) and `complete()` (lines 104-115, ✅ fully fixed)

**Note:** Verified file is `src/audioengine/CrossfadeRuntime.h` (NOT `src/audioengine/ISR/CrossfadeRuntime.h` as the bug report states — no `ISR/` subdirectory exists).

**Status (2026-08-09):** **95% FIXED in work88** — see "Major Update" section above.

**Remaining omissions (only in `start()`, only 2 lines):**

**Missing from `start()` (h:38-55):**
- `firstIrDryDone_` — NOT reset (should be `false`)
- `dryScaleTarget_` — NOT reset (should be `1.0`)

> ❌ `dryScaleGain_.setCurrentAndTargetValue(1.0)` is **DELIBERATELY OMITTED** from `start()` — see h:41-42 comment: NonRT→LinearRamp race avoidance. Fade-in is driven by RT-side `armCrossfadeIfPending` at `AudioEngine.h:3887` via `setTargetValue(1.0)`.

**`complete()` (h:104-115): ✅ ALL FIXED in work88** — resets `dryScaleTarget_=1.0` (h:106), `startDelayBlocks_=0` (h:107), `dryHoldSamples_=0` (h:108), `pending_=false`, `useDryAsOld_=false`, `firstIrDryPending_=false`, `firstIrDryDone_=false`, `queuedFadeTimeSec_=0.030`, `fadeStartTimestampUs_=0`, + `bumpCrossfadeGeneration()` (h:120).

> ❌ `dryScaleGain_.setCurrentAndTargetValue(1.0)` is **DELIBERATELY OMITTED** from `complete()` — see h:121 comment: NonRT→LinearRamp race avoidance.

**LATENT BUG NOTE (verified 2026-08-09):** `setDryScaleTarget()` (`src/audioengine/CrossfadeRuntime.h:180`) has **zero callers** across all `.cpp`/`.h` files. The field `dryScaleTarget_` is only ever written by:
1. Constructor initialization: `std::atomic<double> dryScaleTarget_{ 1.0 };` (h:206)
2. `reset()`: `convo::publishAtomic(dryScaleTarget_, 1.0, ...)` (h:110)
3. `complete()`: `convo::publishAtomic(dryScaleTarget_, 1.0, ...)` (h:136) — added in work88

This means `dryScaleTarget_` is always `1.0` in the current codebase, making the stale-value scenario **latent** (not currently triggered). However, `setDryScaleTarget()` exists as a public API — if any future code calls it with a non-1.0 value, the stale state will immediately become a live bug. The fix (resetting `dryScaleTarget_` in `start()` as well) is still correct and necessary for defensive correctness — this is 1 of the 2 remaining atomic publishes needed.

**Workaround NOW REDUNDANT (work88 made `complete()` self-sufficient):** The Timer path in `AudioEngine.Timer.cpp:896-898` manually resets `startDelayBlocks_` and `dryHoldSamples_` after `complete()` — this was a pre-work88 workaround. After work88, `complete()` ALREADY resets both fields internally (h:107-108), making these manual workarounds redundant. The second `complete()` call at `AudioEngine.Timer.cpp:1580` only manually calls `setDryHoldSamples(0)` at :1583 — `startDelayBlocks_` is NOT manually reset there, but is now handled by `complete()` internally.
```cpp
// AudioEngine.Timer.cpp:896-898 (pre-work88 workaround, NOW redundant):
crossfadeRuntime_.complete();
crossfadeRuntime_.setStartDelayBlocks(0);  // redundant — complete() resets at h:107
crossfadeRuntime_.setDryHoldSamples(0);    // redundant — complete() resets at h:108
```
But this workaround only covered the Timer path — the `DSPTransition.h:66,126` `complete()` calls did NOT have these manual resets, leaving `startDelayBlocks_`, `dryHoldSamples_` stale on those paths before work88. After work88, `complete()` resets both fields internally (h:107-108), so ALL 4 complete() call sites now correctly reset these fields. The Timer path workaround can be removed (defensive redundancy) but should be kept during transition.

**Complete() call sites (verified):**
- `src/audioengine/AudioEngine.Timer.cpp:896` — Timer (NonRT) — has manual workaround for `startDelayBlocks_` + `dryHoldSamples_`
- `src/audioengine/AudioEngine.Timer.cpp:1580` — Timer (NonRT) — only manually calls `setDryHoldSamples(0)` at :1583; `startDelayBlocks_` NOT manually reset
- `src/audioengine/DSPTransition.h:66` — Emergency override path (NonRT via `onPublishCompleted`)
- `src/audioengine/DSPTransition.h:126` — Immediate retire path (NonRT via `onPublishCompleted`)

All callers are NonRT. However, the RT Audio Thread **concurrently reads** `dryScaleGain_` via `getDryScaleGain().getNextValue()` (AudioBlock.cpp:442, BlockDouble.cpp:421) during `start()`/`complete()` execution. `LinearRamp`'s `current/target/step/remaining` members are NON-atomic, so writing them from NonRT while RT reads is a data race. This is why work88 DELIBERATELY omits `dryScaleGain_.setCurrentAndTargetValue(1.0)` from both methods. The fade-in is driven by RT-side `armCrossfadeIfPending` via `setTargetValue(1.0)` instead.

**Impact (post-work88, before the 2-line `start()` fix):**
- RT path `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:442`: `crossfadeRuntime_.getDryScaleGain().getNextValue()` — RT-driven via `armCrossfadeIfPending`, no longer stale after complete() (work88 removed `dryScaleGain_` write from both methods)
- RT path `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:421`: same — RT-driven
- `src/audioengine/AudioEngine.h:2897`: `getDryScaleTarget()` — stale ONLY if `start()` is called without the 2-line fix and a previous cycle set `dryScaleTarget_` to non-1.0 (currently latent since `setDryScaleTarget()` has 0 callers)
- `src/audioengine/AudioEngine.h:2998`: same stale-read path during snapshot build

**Fix (corrected — 2026-08-09):**
```cpp
// In start() — add after fadeStartTimestampUs_ publish (h:49), before bumpCrossfadeGeneration() (h:58):
convo::publishAtomic(firstIrDryDone_, false, std::memory_order_release);
convo::publishAtomic(dryScaleTarget_, 1.0, std::memory_order_release);

// In complete() — NO CHANGES NEEDED (already fixed in work88 at h:106-108).
```

> **❌ Previous plan ERRATA (corrected):** The original plan proposed adding `dryScaleGain_.setCurrentAndTargetValue(1.0)` to both methods. This is **WRONG** and must NOT be applied. Both methods deliberately omit `dryScaleGain_` writes due to the NonRT→LinearRamp race (h:41-42 and h:121 comments). Adding it would re-introduce the data race that work88 specifically removed.

**Files to modify:** `src/audioengine/CrossfadeRuntime.h`

---

## 📋 Fix Summary Table

| Bug | Priority | Fix Description | Files to Modify | RT-Safe? | Risk |
|-----|----------|-----------------|-----------------|----------|------|
| BUG-014 | P0 | Replace `juce::String currentDeviceTypeName_` (h:2355) with `std::atomic<const char*>`; use `strstr` instead of `containsIgnoreCase`; `setAudioDeviceTypeName` uses `new char[]`/`delete[]` (MSVC-compatible, no `strdup`); remove `getAudioDeviceTypeName()` (0 callers); add cleanup in `~AudioEngine` | `AudioEngine.h:2347-2355`, `AudioEngine.Mmcss.cpp:50-64`, `AudioEngine.CtorDtor.cpp:89` | Yes | Low — pure swap, no logic change |
| BUG-015 | — | **ALREADY FIXED** (see Already Fixed table). Only remaining TODO: `ISRRetireRouter::retire` cpp:154 — RuntimeHealthMonitor notification (observability only, not recovery). **NOT a correctness bug.** | — | — | — |
| BUG-027 | — | **ALREADY FIXED** (see Already Fixed table). `completeFade` recovery via `quarantineRetireSink`; `updateFade` deleted; race no longer applicable. | — | — | — |
| BUG-028 | P1 | **95% FIXED (work88)** — `complete()` ALREADY resets `dryScaleTarget_=1.0`, `startDelayBlocks_=0`, `dryHoldSamples_=0` + `bumpCrossfadeGeneration()`. `start()` ALREADY removed `gain_.setCurrentAndTargetValue(0.0)` (race avoidance). **Remaining: add 2 atomic publishes to `start()` only**: `firstIrDryDone_=false`, `dryScaleTarget_=1.0`. ❌ DO NOT add `dryScaleGain_.setCurrentAndTargetValue(1.0)` — deliberately omitted (h:121 NonRT→LinearRamp race). | `CrossfadeRuntime.h:50-51` (add 2 lines in `start()`) | Yes (NonRT) | Low — 2 atomic publishes |

---

### BUG-014 Fix (step-by-step)

**Step 1 — `AudioEngine.h:2347-2355`** (member at h:2355, setter at h:2347, getter at h:2348):

```cpp
// Before (AudioEngine.h:2347-2355):
void setAudioDeviceTypeName(const juce::String& type) noexcept { currentDeviceTypeName_ = type; }
[[nodiscard]] const juce::String& getAudioDeviceTypeName() const noexcept { return currentDeviceTypeName_; }
// ...
juce::String currentDeviceTypeName_;

// After (AudioEngine.h:2347-2355):
alignas(64) std::atomic<const char*> currentDeviceTypeNameRaw_{nullptr};  // h:2355

void setAudioDeviceTypeName(const juce::String& type) noexcept {
    const char* old = currentDeviceTypeNameRaw_.load(std::memory_order_acquire);
    // MSVC-compatible allocation: strdup is POSIX-only, not available in MSVC 19.44+
    // Use new[]/delete[] — consistent with JUCE's StringHolder (createUninitialisedBytes uses new char[])
    const int numBytes = type.getNumBytesAsUTF8();
    char* newStr = new char[static_cast<size_t>(numBytes) + 1];
    memcpy(newStr, type.toRawUTF8(), static_cast<size_t>(numBytes) + 1);
    currentDeviceTypeNameRaw_.store(newStr, std::memory_order_release);
    if (old) delete[] old;
}
// getAudioDeviceTypeName() has 0 callers (verified via rg) — remove entirely.
// getCurrentMmcppPolicy() in AudioEngine.Mmcss.cpp loads from the atomic instead.
```

> **Note:** `juce::String::toRawUTF8()` returns a pointer valid only for the lifetime of the `juce::String` parameter. `memcpy` into `new char[]` makes a persistent copy. `alignas(64)` prevents false sharing. `strstr` (used in Step 2) IS available on MSVC via `<cstring>`. The destructor at `AudioEngine.CtorDtor.cpp:89` must `delete[]` the pointer.

**Step 2 — `AudioEngine.Mmcss.cpp:50-64`:** Replace `const auto& type = currentDeviceTypeName_` + `type.containsIgnoreCase(...)` with `const char* typeRaw = currentDeviceTypeNameRaw_.load(std::memory_order_acquire)` + `strstr(typeRaw, "ASIO")` etc.

**Step 3 — `AudioEngine.CtorDtor.cpp:89` (destructor):** Add `const char* name = currentDeviceTypeNameRaw_.load(std::memory_order_acquire); if (name) delete[] name;` to prevent memory leak on shutdown.

#### Detailed Design — BUG-014

**Design Rationale:** `juce::String` uses Copy-On-Write (CoW) with `std::atomic<int>` refCount, but the `text` pointer member itself is non-atomic. The RT thread reads `text` via `containsIgnoreCase()` without `retain()`, creating a use-after-free window when the NonRT thread's `operator=` swaps in a new string and frees the old buffer.

**Fix Approach:** Replace the entire `juce::String` member with a raw `std::atomic<const char*>`. The atomic pointer swap guarantees the RT thread never reads a half-written state — it either sees the old pointer or the new one, never garbage.

**Memory Ordering:**
- **Store path** (`setAudioDeviceTypeName`, NonRT/Message Thread): `load(acquire)` old, `new char[]`+memcpy new, `store(release)` new, then `delete[] old` — the release store ensures the allocation+memcpy + store is ordered before the `delete[]`; the acquire load ensures we see the latest committed pointer before we attempt to free it
- **Load path** (`getCurrentMmcppPolicy`, RT/Audio Thread): `load(acquire)` raw pointer → pass to `strstr` — single atomic read, no retain/release needed because the pointer is either valid (old or new) and will never be freed from under us (the release store in `setAudioDeviceTypeName` guarantees the `delete[] old` happens-after any `acquire` load that already saw the old pointer)

**Ownership Semantics:**
- `setAudioDeviceTypeName` owns: it `new[]`es on write, `delete[]`s the previous pointer on overwrite
- `getCurrentMmcppPolicy` borrows: it only reads `strstr`, never frees — it holds a valid pointer snapshot for the duration of the function
- `~AudioEngine` destructor owns final cleanup: loads the pointer and `delete[]`s it

**Fallback Handling:** If `new char[]` returns `nullptr` (OOM — extremely rare since it would throw, not return null), the old pointer is left in place (no swap, no delete). RT thread continues with the old device type — degraded but safe. Note: `new char[n]` throws `std::bad_alloc` on failure by default; in an audio engine context, this is caught at a higher level. For strict `noexcept` compliance, could use `new(std::nothrow) char[n]`.

**Thread Safety:** ✅ RT-safe — single `std::atomic<const char*>::load(acquire)` on the RT path, no allocations, no virtual calls. `strstr` is a plain C function, no heap access.

**Testing Approach:**
- TSan/helgrind stress test: spawn NonRT thread hammering `setAudioDeviceTypeName` while RT thread calls `getCurrentMmcppPolicy` in a tight loop — verify zero data races
- OOM simulation: mock `new char[]` to throw — verify old pointer is preserved, no crash
- Correctness: verify `strstr` matching produces identical `MmcppPolicy` results as the old `containsIgnoreCase` approach
- Destructor: verify no memory leak at shutdown (check `delete[]` is called in `~AudioEngine`)

---

### BUG-028 Fix (step-by-step) — 95% Already Applied, Only 2 Lines Remaining

**Current state (verified 2026-08-09):** `complete()` (h:104-115) ALREADY resets all 9 atomic fields + `bumpCrossfadeGeneration()`. `start()` (h:38-55) resets 7 of 9 fields. Only 2 atomic publishes are missing from `start()`.

**`start()` — add after `fadeStartTimestampUs_` publish (h:49), before `bumpCrossfadeGeneration()`:**
```cpp
convo::publishAtomic(firstIrDryDone_, false, std::memory_order_release);
convo::publishAtomic(dryScaleTarget_, 1.0, std::memory_order_release);
```

**`complete()` — NO CHANGES NEEDED (already fixed in work88).**

> **❌ DO NOT add `dryScaleGain_.setCurrentAndTargetValue(1.0)` to either method.**
> - `start()`: The old `gain_.setCurrentAndTargetValue(0.0)` was DELIBERATELY REMOVED (h:41-42 comment) — fade-in is driven by RT-side `armCrossfadeIfPending` at `AudioEngine.h:3887`.
> - `complete()`: `dryScaleGain_.setCurrentAndTargetValue(1.0)` is DELIBERATELY OMITTED (h:121 comment) — NonRT→LinearRamp race avoidance.
> - Adding `setCurrentAndTargetValue` to NonRT `start()`/`complete()` while RT reads `getDryScaleGain().getNextValue()` is a data race (LinearRamp members are non-atomic).
> - `reset()` (shutdown-only, h:127-139) DOES call `dryScaleGain_.setCurrentAndTargetValue(1.0)` — correct because Audio Thread is stopped during `reset()`.

#### Detailed Design — BUG-028 (Updated 2026-08-09 — 95% Already Fixed)

**Design Rationale:** `CrossfadeRuntime` maintains several fields that control the dry signal mixing path. `reset()` resets all fields but is only called at shutdown. `start()` and `complete()` — called at the beginning and end of each fade cycle — must reset the dry-scale fields to prevent stale values from persisting into the next cycle.

**Fields Requiring Reset in `start()` (only 2 remaining — work88 already fixed 7):**
| Field | Type | Old Value (from prior cycle) | New Value | Status | Why |
|-------|------|------------------------------|-----------|--------|-----|
| `firstIrDryDone_` | `bool` atomic | `true` (previous cycle completed) | `false` | ❌ MISSING | Ensures IR dry path is re-evaluated for first input |
| `dryScaleTarget_` | `double` atomic | Stale gain target | `1.0` | ❌ MISSING | Default: dry signal at full volume |
| `dryScaleGain_` | `LinearRamp` | Stale interpolated value | `1.0` | 🚫 DO NOT FIX | NonRT→LinearRamp race (h:121). RT-side `armCrossfadeIfPending` drives fade-in instead. |

**Fields Requiring Reset in `complete()` — ✅ ALL ALREADY FIXED in work88:**
| Field | Type | Old Value | New Value | Status | Why |
|-------|------|-----------|-----------|--------|-----|
| `dryScaleTarget_` | `double` atomic | Fade-end target | `1.0` | ✅ h:106 | Post-fade: dry at full volume |
| `startDelayBlocks_` | `int` atomic | Delay applied during fade | `0` | ✅ h:107 | No delay after fade completion |
| `dryHoldSamples_` | `int` atomic | Samples held during IR loading | `0` | ✅ h:108 | No hold after completion |
| `dryScaleGain_` | `LinearRamp` | Faded gain value | `1.0` | 🚫 DO NOT FIX | NonRT→LinearRamp race (h:121). RT consumes via `getNextValue()` acquire. |

**RT-Safety:** All callers of `start()` and `complete()` are NonRT:
- `start()`: Called from `AudioEngine.Timer.cpp` (Message Thread) and `DSPTransition.h:66` (emergency override, NonRT)
- `complete()`: Called from `AudioEngine.Timer.cpp:896,1580` (Message Thread) and `DSPTransition.h:66,126` (NonRT)

> **❌ `dryScaleGain_.setCurrentAndTargetValue(1.0)` is NOT safe in `start()`/`complete()`.** Although callers are NonRT, the RT Audio Thread concurrently reads `getDryScaleGain().getNextValue()` (AudioBlock.cpp:442, BlockDouble.cpp:421). `LinearRamp`'s `current/target/step/remaining` members are NON-atomic. Writing them from NonRT while RT reads is a data race. This is why work88 DELIBERATELY removed `gain_.setCurrentAndTargetValue(0.0)` from `start()` (h:41-42) and DELIBERATELY omits `dryScaleGain_.setCurrentAndTargetValue(1.0)` from `complete()` (h:121).
>
> `reset()` (shutdown-only, h:127-139) DOES call `dryScaleGain_.setCurrentAndTargetValue(1.0)` — correct because Audio Thread is stopped during `reset()`.

**`bumpCrossfadeGeneration()` (NEW in work88, h:193):** Increments `crossfadeGeneration_` atomic counter with `release` semantics. Called at end of `start()` (h:58) and `complete()` (h:120). Purpose: "block-boundary semantic consistency anchor" — allows RT to detect that a coherent batch of atomic publishes has completed (single atomic read of generation = consistent snapshot). Referenced as "BUG-028 五次レビュー §8" (5th review section 8) in comments.

**Memory Ordering:** `convo::publishAtomic` uses `memory_order_release` — appropriate because the Audio Thread reads these via `convo::consumeAtomic` (acquire) before the fade cycle begins. The `start()`/`complete()` writes are sequenced before the audio thread's first read in the next cycle.

**Interaction with Existing Workaround:** The Timer path at `AudioEngine.Timer.cpp:896-898` already manually calls `setStartDelayBlocks(0)` + `setDryHoldSamples(0)` after `complete()`. Once `complete()` internally resets these fields, the manual workaround becomes redundant. However, the second `complete()` call at `AudioEngine.Timer.cpp:1580` only calls `setDryHoldSamples(0)` (line 1583) — `startDelayBlocks_` is NOT reset there. The fix in `complete()` will cover both Timer paths. The manual workaround lines can be removed after verifying the fix, but should be kept as defensive redundancy during transition.

**Testing Approach:**
- Unit test: call `start()` then immediately check all fields via `getXxx()` — verify `firstIrDryDone_==false`, `dryScaleTarget_==1.0`. (❌ Do NOT assert `dryScaleGain_.getCurrentValue()` — it's RT-driven via `armCrossfadeIfPending`.)
- Unit test: call `complete()` then check all fields — verify `dryScaleTarget_==1.0`, `startDelayBlocks_==0`, `dryHoldSamples_==0` (all ALREADY reset by work88). (❌ Do NOT assert `dryScaleGain_`.)
- Integration test: trigger multiple fade cycles (start → complete → start → complete) and verify atomic fields are correctly reset at each cycle boundary
- Regression test: verify Timer path behavior unchanged — work88 `complete()` already resets `startDelayBlocks_`/`dryHoldSamples_` internally, so manual workaround at Timer.cpp:896-898 is now redundant (can be removed after verification)
- TSan test: verify NO data race on `dryScaleGain_` when NonRT `start()`/`complete()` runs concurrently with RT `getDryScaleGain().getNextValue()` — the deliberate omission of `setCurrentAndTargetValue` should eliminate the race

---

## ❓ Missing File

| Bug | Status |
|-----|--------|
| BUG-017 | `BUG-017.md` does not exist in `doc/work88/mini_bugs_unchecked/` — directory goes BUG-016 → BUG-018. No git history entry for BUG-017 found. Likely skipped during bug report generation. |

---

## Files Modified (Already Applied)
- `CMakeLists.txt` — guarded MKL test links with `CONVOPEQ_HAS_MKL`
- `.github/workflows/isr-authority-compliance.yml` — added `-DCONVOPEQ_REQUIRE_MKL=OFF`
- `.github/workflows/sanitizer-ci.yml` — replaced deprecated intel setup with `choco install intel-oneapi-basekit`
- `.github/workflows/soak-ci.yml` — same replacement
- `src/MainWindow.cpp` — latency display fix (`hasActiveRuntimeDSP()` guard)

## Actionable Todo

1. [ ] BUG-014: (a) Replace `juce::String` member with `std::atomic<const char*>` + `new[]`/`delete[]` in `AudioEngine.h:2347-2355`; (b) Replace `containsIgnoreCase` with `strstr` in `AudioEngine.Mmcss.cpp:50-64`; (c) Remove `getAudioDeviceTypeName()` (0 callers); (d) Add `delete[]` cleanup in `~AudioEngine` at `AudioEngine.CtorDtor.cpp:89`
2. [ ] BUG-015: ~~All 10 sites fixed via `RetireQuarantineStore` pattern~~ ✅ DONE. Remaining cosmetic TODO: `ISRRetireRouter::retire` cpp:154 — replace `// ★ Future: RuntimeHealthMonitor へ通知` with actual `engineHealthMonitor_.notify(RetireEnqueueResult)` call (observability only, NOT a correctness bug)
3. [ ] BUG-027: ~~CompleteFade path fixed via `quarantineRetireSink`~~ ✅ DONE. No remaining work.
4. [ ] BUG-028: Add 2 missing atomic publishes to `CrossfadeRuntime::start()` in `src/audioengine/CrossfadeRuntime.h` (after `fadeStartTimestampUs_` publish at h:49, before `bumpCrossfadeGeneration()` at h:58):
   - `convo::publishAtomic(firstIrDryDone_, false, std::memory_order_release);`
   - `convo::publishAtomic(dryScaleTarget_, 1.0, std::memory_order_release);`
   - ❌ DO NOT add `dryScaleGain_.setCurrentAndTargetValue(1.0)` — deliberately omitted (NonRT→LinearRamp race, h:121). `complete()` is ALREADY 100% fixed in work88 — no changes needed there.
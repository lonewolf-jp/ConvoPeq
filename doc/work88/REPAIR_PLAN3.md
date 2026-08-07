# REPAIR PLAN 3 — Verified Bug Report Analysis

## Date: 2026-08-06
## Scope: All 36 bug reports in `doc/work88/mini_bugs_unchecked/BUG-0XX.md` (BUG-017 file is missing from disk)
## Method: Source code verification (rg, grep, sed, fdfind, WSL) + Web research (JUCE source, GitHub)

---

## Executive Summary

| Status | Count | Bugs |
|--------|-------|------|
| ✅ Already Fixed | 31 | 011-013, 016, 018-026, 029-046 |
| 🟡 Partially Fixed | 2 | BUG-015, BUG-027 |
| 🔴 Still Open | 2 | BUG-014, BUG-028 |
| ❓ Missing/Not Applicable | 1 | BUG-017 (file absent from disk) |

> Total bug report files on disk: 35 (BUG-011 through BUG-046, excluding BUG-017).

---

## ✅ Already Fixed (31 bugs)

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

---

## 🟡 Partially Fixed (2 bugs)

**IMPORTANT — Two Different `enqueueWithRetry` Functions:** There are **two distinct** `enqueueWithRetry` functions in the codebase:

- **`SnapshotCoordinator::enqueueWithRetry`** (static helper, `SnapshotCoordinator.h:134-156`) — returns `bool`. Has internal `tryReclaim + 1 retry` logic.
- **`ISRRetireRouter::enqueueWithRetry`** (member function, `ISRRetireRouter.h:96-98`) — returns `RetireEnqueueResult` (enum). Has its OWN internal `tryReclaim + 2 retries` logic.

| Bug | File:Line | What Was Done | What's Missing |
|-----|-----------|---------------|-----------------|
| BUG-015 | `src/core/SnapshotCoordinator.cpp:38` (startFade) `src/core/SnapshotCoordinator.cpp:94` (completeFade) `src/core/SnapshotCoordinator.h:100` (switchImmediate oldSnap) `src/core/SnapshotCoordinator.h:158,160` (retireCurrentAndTarget) — all call **`SnapshotCoordinator::enqueueWithRetry`** (static, returns `bool`) | Return value captured at cpp:38, cpp:94, h:88. NOT captured at h:100, h:158, h:160. All failure handlers are `// ★ Future: RuntimeHealthMonitor へ通知` TODO-only. | 1. **h:100** — `enqueueWithRetry` for `oldSnap` in `switchImmediate()` — return NOT captured. 2. **h:158, h:160** — `retireCurrentAndTarget()` — return NOT captured at either site. 3. **All 6 SnapshotCoordinator sites** lack actual recovery logic (TODO only). |
| BUG-015 | `src/audioengine/ISRRetireRouter.cpp:154` (retire) `src/audioengine/DSPLifetimeManager.cpp:49,90` — both call **`ISRRetireRouter::enqueueWithRetry`** (member, returns `RetireEnqueueResult`) | ISRRetireRouter::retire (cpp:154) captures return and checks `if (result != Success)`. DSPLifetimeManager.cpp:49,90 capture into `result` but `juce::ignoreUnused(result)` discards it. | 1. **ISRRetireRouter.cpp:154** failure handler is TODO-only (no direct delete fallback). 2. **DSPLifetimeManager.cpp:49,90** — result discarded via `ignoreUnused`; need `if (result != Success) { deleter(ptr); }`. Note: `ISRRetireRouter::enqueueWithRetry` already has internal retry (tryReclaim ×2, 2 retries), so only direct delete is needed — NOT additional retry. |
| BUG-027 | `src/core/SnapshotCoordinator.cpp:94` (completeFade) | Same site as BUG-015. `updateFade()` DELETED (BUG-031). `completeFade` captures return, failure handler is TODO. | Same as BUG-015: replace TODO with direct delete fallback. Note: BUG-027 bug report describes a race between `completeFade` and `updateFade`, but `updateFade` is deleted — race no longer applicable. Remaining issue is failure handler only. |

### Root Cause Analysis (BUG-015)

There are two distinct `enqueueWithRetry` functions with different recovery needs:

**Category A — `SnapshotCoordinator::enqueueWithRetry` (static helper, h:134-156):** Returns `bool`. Internally does `enqueueRetire → tryReclaim → enqueueRetire → return false`. Used at `SnapshotCoordinator.cpp:38` (startFade), `SnapshotCoordinator.cpp:94` (completeFade), `SnapshotCoordinator.h:88` (switchImmediate oldTarget — already captured), `SnapshotCoordinator.h:100` (switchImmediate oldSnap — NOT captured), `SnapshotCoordinator.h:158,160` (retireCurrentAndTarget — NOT captured). When it returns `false`, all internal retries are exhausted — the pointer is leaked.

**Category B — `ISRRetireRouter::enqueueWithRetry` (member function, `ISRRetireRouter.h:96`):** Returns `RetireEnqueueResult` (enum: Success/QueuePressure/QueueFull/Shutdown). Internally does `enqueueRetire → tryReclaim + 2 retries`. Used at `ISRRetireRouter.cpp:154` (retire — return already captured, TODO handler only) and `DSPLifetimeManager.cpp:49,90` (return captured but `juce::ignoreUnused(result)` — discarded). When it returns non-Success, all internal retries are exhausted — the pointer is leaked.

**Key distinction:** `ISRRetireRouter::enqueueWithRetry` already performs `tryReclaim + retry` internally. Adding more `tryReclaim + retry` at the call site is REDUNDANT. The correct fix for Category B sites is simply: `if (result != RetireEnqueueResult::Success) { deleter(ptr); }` (direct delete, since caller is NonRT).

**`RetireEnqueueResult` semantics (from `ISRAuthorityClass.h:25-30`):**
- `Success` (0): enqueued into deferred deletion queue
- `QueuePressure`: queue full, retried, still failed → pointer NOT enqueued
- `QueueFull`: fallback depth exceeded → pointer IS enrolled via fallback (do NOT direct delete)
- `Shutdown`: system shutting down → pointer was rejected (do NOT direct delete, log only)

Currently only `Success` and `QueuePressure` are returned by `ISRRetireRouter::enqueueRetire`. `QueueFull` and `Shutdown` are forward-declared but not yet produced. The fix handles `QueuePressure` with direct delete; `QueueFull`/`Shutdown` should only be logged until the enqueue path supports them.

### EnqueueWithRetry Implementation (verified — two functions)

```cpp
// src/core/SnapshotCoordinator.h:134-156 — static helper (Category A)
static bool enqueueWithRetry(convo::IEpochProvider& provider,
                             void* ptr, void (*deleter)(void*),
                             uint64_t epoch) noexcept {
    if (provider.enqueueRetire(ptr, deleter, epoch))   // ← ISRRetireRouter::enqueueRetire (bool overload) internally tries tryReclaim+retry
        return true;
    provider.tryReclaim();
    if (provider.enqueueRetire(ptr, deleter, epoch))
        return true;
    return false;  // ← Pointer leaked! All retries exhausted.
}

// src/audioengine/ISRRetireRouter.cpp:159-183 — member function (Category B)
RetireEnqueueResult ISRRetireRouter::enqueueWithRetry(void* ptr,
                                                        void (*deleter)(void*),
                                                        uint64_t epoch,
                                                        DeletionEntryType type) noexcept
{
    auto result = enqueueRetire(ptr, deleter, epoch, type);    // internal tryReclaim+retry with 500ms cooldown
    if (result == RetireEnqueueResult::Success) return result;
    constexpr int kMaxRetry = 2;
    for (int attempt = 0; attempt < kMaxRetry; ++attempt) {
        provider_->tryReclaim();
        result = enqueueRetire(ptr, deleter, epoch, type);
        if (result == RetireEnqueueResult::Success) return result;
        if (result != RetireEnqueueResult::QueuePressure) break;
    }
    return RetireEnqueueResult::QueuePressure;  // ← Pointer leaked! Caller must handle.
}
```

### Fix Needed (BUG-015 + BUG-027)

**Category A — `SnapshotCoordinator::enqueueWithRetry` sites (static, returns `bool`):**

1. `SnapshotCoordinator.h:100` (switchImmediate oldSnap) — capture return: `const auto result = enqueueWithRetry(...)`, add `if (!result) { provider.tryReclaim(); if (!enqueueWithRetry(...)) { deleter(ptr); } }`
2. `SnapshotCoordinator.h:158` (retireCurrentAndTarget current) — same pattern
3. `SnapshotCoordinator.h:160` (retireCurrentAndTarget target) — same pattern
4. `SnapshotCoordinator.cpp:38` (startFade) — replace `// ★ Future: RuntimeHealthMonitor へ通知` with recovery
5. `SnapshotCoordinator.cpp:94` (completeFade, also BUG-027) — replace `// ★ Future: RuntimeHealthMonitor へ通知` with recovery
6. `SnapshotCoordinator.h:88` (switchImmediate oldTarget) — replace `// ★ Future: RuntimeHealthMonitor へ通知` with recovery

**Category B — `ISRRetireRouter::enqueueWithRetry` sites (member, returns `RetireEnqueueResult`):**

7. `ISRRetireRouter.cpp:154` (retire) — replace `// ★ Future: RuntimeHealthMonitor へ通知` with `if (result == RetireEnqueueResult::QueuePressure) { deleter(ptr); }` (direct delete only — `enqueueWithRetry` already retried internally)
8. `DSPLifetimeManager.cpp:49` — remove `juce::ignoreUnused(result)`, add `if (result != RetireEnqueueResult::Success) { deleter(dsp); }`
9. `DSPLifetimeManager.cpp:90` — remove `juce::ignoreUnused(result)`, add `if (result != RetireEnqueueResult::Success) { deleter(toDelete); }`

> **Note on BUG-027:** `completeFade` at `SnapshotCoordinator.cpp:94` is a Category A site. The BUG-027 bug report describes a race between `completeFade` and `updateFade`, but `updateFade` was deleted as dead code (BUG-031). The race scenario is no longer possible. The only remaining issue at this site is the TODO failure handler — identical to BUG-015.

---

## 🔴 Still Open (2 bugs)

### BUG-014: juce::String data race on `currentDeviceTypeName_`

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

### BUG-028: CrossfadeRuntime.start()/complete() incomplete reset

**Severity:** HIGH (audio artifact risk — stale dry scale gain on RT path)

**Location:** `src/audioengine/CrossfadeRuntime.h` — `start()` (lines 38-51) and `complete()` (lines 95-103)

**Note:** Verified file is `src/audioengine/CrossfadeRuntime.h`.

**Root Cause:**
The bug report originally identified that `complete()` misses resetting `useDryAsOld_`, `firstIrDryPending_`, and `firstIrDryDone_`. The current code shows that `complete()` **DOES** reset all three of those fields (lines 98-100) — those are FIXED.

However, NEW omissions remain in both methods:

**Missing from `start()` (h:38-51):**
- `firstIrDryDone_` — NOT reset (should be `false`)
- `dryScaleTarget_` — NOT reset (should be `1.0`)
- `dryScaleGain_` (LinearRamp) — NOT reset to `1.0`

**Missing from `complete()` (h:95-103):**
- `dryScaleTarget_` — NOT reset (should be `1.0`)
- `dryScaleGain_` (LinearRamp) — NOT reset to `1.0`
- `startDelayBlocks_` — NOT reset (should be `0`)
- `dryHoldSamples_` — NOT reset (should be `0`)

**LATENT BUG NOTE (verified 2026-08-06):** `setDryScaleTarget()` (`src/audioengine/CrossfadeRuntime.h:153`) has **zero callers** across all `.cpp`/`.h` files. The field `dryScaleTarget_` is only ever written by:
1. Constructor initialization: `std::atomic<double> dryScaleTarget_{ 1.0 };` (h:171)
2. `reset()`: `convo::publishAtomic(dryScaleTarget_, 1.0, ...)` (h:115)

This means `dryScaleTarget_` is always `1.0` in the current codebase, making the stale-value scenario **latent** (not currently triggered). However, `setDryScaleTarget()` exists as a public API — if any future code calls it with a non-1.0 value, the stale state will immediately become a live bug. The fix (resetting in `start()`/`complete()`) is still correct and necessary for defensive correctness.

**Workaround already in place (partial):** The Timer path in `AudioEngine.Timer.cpp:896-898` manually resets `startDelayBlocks_` and `dryHoldSamples_` after `complete()` (but the Timer path does NOT reset `dryScaleTarget_` or `dryScaleGain_`). The second `complete()` call at `AudioEngine.Timer.cpp:1580` only manually calls `setDryHoldSamples(0)` at :1583 — `startDelayBlocks_` is NOT reset there either.
```cpp
crossfadeRuntime_.complete();
crossfadeRuntime_.setStartDelayBlocks(0);
crossfadeRuntime_.setDryHoldSamples(0);
```
But this workaround only covers the Timer path — the `DSPTransition.h:66,126` `complete()` calls do NOT have these manual resets, leaving `startDelayBlocks_`, `dryHoldSamples_`, `dryScaleTarget_`, and `dryScaleGain_` stale on those paths.

**Complete() call sites (verified):**
- `src/audioengine/AudioEngine.Timer.cpp:896` — Timer (NonRT) — has manual workaround for `startDelayBlocks_` + `dryHoldSamples_`
- `src/audioengine/AudioEngine.Timer.cpp:1580` — Timer (NonRT) — only manually calls `setDryHoldSamples(0)` at :1583; `startDelayBlocks_` NOT manually reset
- `src/audioengine/DSPTransition.h:66` — Emergency override path (NonRT via `onPublishCompleted`)
- `src/audioengine/DSPTransition.h:126` — Immediate retire path (NonRT via `onPublishCompleted`)

All callers are NonRT, so `dryScaleGain_.setCurrentAndTargetValue(1.0)` is safe from an RT-safety perspective. The issue is purely about stale values persisting across fade cycles.

**Impact:**
- RT path `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:442`: `crossfadeRuntime_.getDryScaleGain().getNextValue()` returns stale gain
- RT path `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:421`: same stale gain read
- `src/audioengine/AudioEngine.h:2897`: `getDryScaleTarget()` returns stale target
- `src/audioengine/AudioEngine.h:2998`: `getDryScaleTarget()` returns stale target during snapshot build

**Fix:**
```cpp
// In start() — add after fadeStartTimestampUs_ publish (h:49):
convo::publishAtomic(firstIrDryDone_, false, std::memory_order_release);
convo::publishAtomic(dryScaleTarget_, 1.0, std::memory_order_release);
dryScaleGain_.setCurrentAndTargetValue(1.0);

// In complete() — add after fadeStartTimestampUs_ publish (h:102):
convo::publishAtomic(dryScaleTarget_, 1.0, std::memory_order_release);
convo::publishAtomic(startDelayBlocks_, 0, std::memory_order_release);
convo::publishAtomic(dryHoldSamples_, 0, std::memory_order_release);
dryScaleGain_.setCurrentAndTargetValue(1.0);
```

**Files to modify:** `src/audioengine/CrossfadeRuntime.h`

---

## 📋 Fix Summary Table

| Bug | Priority | Fix Description | Files to Modify | RT-Safe? | Risk |
|-----|----------|-----------------|-----------------|----------|------|
| BUG-014 | P0 | Replace `juce::String currentDeviceTypeName_` (h:2355) with `std::atomic<const char*>`; use `strstr` instead of `containsIgnoreCase`; `setAudioDeviceTypeName` uses `new char[]`/`delete[]` (MSVC-compatible, no `strdup`); remove `getAudioDeviceTypeName()` (0 callers); add cleanup in `~AudioEngine` | `AudioEngine.h:2347-2355`, `AudioEngine.Mmcss.cpp:50-64`, `AudioEngine.CtorDtor.cpp:89` | Yes | Low — pure swap, no logic change |
| BUG-015 | P1 | **Category A** (SnapshotCoordinator::enqueueWithRetry, returns bool): capture return at h:100, h:158, h:160; replace TODO at cpp:38,94,h:88 with direct delete. **Category B** (ISRRetireRouter::enqueueWithRetry, returns RetireEnqueueResult): replace TODO at ISRRetireRouter.cpp:154; remove `ignoreUnused` at DSPLifetimeManager.cpp:49,90 and add direct delete. Category B needs NO additional retry (enqueueWithRetry already retried internally). | `SnapshotCoordinator.h:88,100,158,160`, `SnapshotCoordinator.cpp:38,94`, `ISRRetireRouter.cpp:154`, `DSPLifetimeManager.cpp:49,90` | Yes (NonRT path) | Medium — 9 sites, 2 patterns |
| BUG-027 | P1 | Same Category A recovery as BUG-015 completeFade path (cpp:94). Note: `updateFade()` was deleted (BUG-031) — race condition no longer applicable. Only failure handler TODO remains. | `SnapshotCoordinator.cpp:94` | Yes | Low — duplicate of BUG-015, 1 site |
| BUG-028 | P1 | In `start()`: publish `firstIrDryDone_=false`, `dryScaleTarget_=1.0`, reset `dryScaleGain_` to 1.0. In `complete()`: publish `dryScaleTarget_=1.0`, `startDelayBlocks_=0`, `dryHoldSamples_=0`, reset `dryScaleGain_` to 1.0 — using `convo::publishAtomic`. Does NOT need `firstIrDryDone_` reset in complete() (already done at h:100). | `CrossfadeRuntime.h:49,102` | Yes (all callers NonRT) | Low — 5 atomic publishes + 1 ramp reset |

### BUG-015 Fix (step-by-step)

**Step 1 — `SnapshotCoordinator::enqueueWithRetry` sites (Category A — static, returns `bool`):**

The `SnapshotCoordinator::enqueueWithRetry` static helper (h:134-156) already performs `enqueueRetire → tryReclaim → enqueueRetire → return false`. When it returns `false`, all internal retries are exhausted. The recovery at the call site should capture the return and do a direct delete on failure. An OPTIONAL extra `tryReclaim + retry` provides belt-and-suspenders safety:

```cpp
// SnapshotCoordinator.h:100 — switchImmediate oldSnap path (NOT captured currently)
// Before:
enqueueWithRetry(*m_epochProvider, oldSnap, snapshotDeleter, newEpoch);

// After:
const auto result = enqueueWithRetry(*m_epochProvider, oldSnap, snapshotDeleter, newEpoch);
if (!result) {
    juce::Logger::writeToLog("SnapshotCoordinator: enqueue failed for oldSnap — direct delete (NonRT)");
    snapshotDeleter(oldSnap);  // or SnapshotCoordinator::deleteSnap(oldSnap)
}
```

```cpp
// SnapshotCoordinator.h:158,160 — retireCurrentAndTarget (NOT captured currently)
// Before:
if (snap) enqueueWithRetry(*m_epochProvider, snap, deleter, retireEpoch);

// After (for both current and target slots):
if (snap) {
    const auto result = enqueueWithRetry(*m_epochProvider, snap, deleter, retireEpoch);
    if (!result) {
        juce::Logger::writeToLog("SnapshotCoordinator: enqueue failed in retireCurrentAndTarget — direct delete (NonRT)");
        deleter(snap);
    }
}
```

**Step 2 — Replace TODO failure handlers at already-captured sites:**

`SnapshotCoordinator.cpp:38` (startFade), `SnapshotCoordinator.cpp:94` (completeFade), `SnapshotCoordinator.h:88` (switchImmediate oldTarget) — all currently have:
```cpp
if (!result) {
    // ★ Future: RuntimeHealthMonitor へ通知
}
```
Replace with:
```cpp
if (!result) {
    juce::Logger::writeToLog("SnapshotCoordinator: enqueue failed — direct delete fallback (NonRT)");
    snapshotDeleter(ptr);  // direct delete — NonRT context, safe
}
```

**Step 3 — `ISRRetireRouter::enqueueWithRetry` sites (Category B — member, returns `RetireEnqueueResult`):**

⚠️ **CRITICAL:** These are a DIFFERENT function. `ISRRetireRouter::enqueueWithRetry` (h:159-183) has its OWN internal retry logic (`tryReclaim + 2 retries`). When it returns non-Success, all internal retries are exhausted. The recovery is simply **direct delete** — NOT additional tryReclaim+retry:

```cpp
// ISRRetireRouter.cpp:154 — retire() (currently captures return, TODO handler)
// Before:
if (result != RetireEnqueueResult::Success) {
    // ★ Future: RuntimeHealthMonitor へ通知
}

// After:
if (result != RetireEnqueueResult::Success) {
    juce::Logger::writeToLog("ISRRetireRouter::retire: queue pressure after internal retries — direct delete (NonRT)");
    deleter(ptr);  // ISRRetireRouter::enqueueWithRetry already retried internally — direct delete is the final fallback
}
```

```cpp
// DSPLifetimeManager.cpp:49 (currently: juce::ignoreUnused(result))
// Before:
const auto result = router_->enqueueWithRetry(dsp, &AudioEngine::destroyDSPCoreNode, epoch, DeletionEntryType::Generic);
juce::ignoreUnused(result);

// After:
const auto result = router_->enqueueWithRetry(dsp, &AudioEngine::destroyDSPCoreNode, epoch, DeletionEntryType::Generic);
if (result != RetireEnqueueResult::Success) {
    juce::Logger::writeToLog("DSPLifetimeManager: enqueue failed — direct delete (NonRT)");
    AudioEngine::destroyDSPCoreNode(dsp);
}
```

```cpp
// DSPLifetimeManager.cpp:90 (same pattern)
const auto result = router_->enqueueWithRetry(toDelete, &AudioEngine::destroyDSPCoreNode, epoch, DeletionEntryType::Generic);
if (result != RetireEnqueueResult::Success) {
    juce::Logger::writeToLog("DSPLifetimeManager: enqueue failed — direct delete (NonRT)");
    AudioEngine::destroyDSPCoreNode(toDelete);
}
```

> **Forward compatibility note:** `RetireEnqueueResult::QueueFull` means the pointer IS enrolled via a fallback path — do NOT direct delete in that case. `RetireEnqueueResult::Shutdown` means the system is shutting down — do NOT direct delete. Currently only `QueuePressure` is returned, so `!= Success → direct delete` is safe. When `QueueFull`/`Shutdown` are implemented, add: `if (result == RetireEnqueueResult::QueuePressure) { deleter(ptr); }`

#### Detailed Design — BUG-015

**Design Rationale:** There are **two distinct** `enqueueWithRetry` functions. `SnapshotCoordinator::enqueueWithRetry` (static, h:134) returns `bool` and does tryReclaim+1 retry internally. `ISRRetireRouter::enqueueWithRetry` (member, h:158) returns `RetireEnqueueResult` and does tryReclaim+2 retries internally. Both can fail after exhausting internal retries, leaking the pointer.

**Recovery Strategy:**
- **Category A (SnapshotCoordinator sites):** Capture `bool` return. On `false`: log + `deleter(ptr)` (direct delete). No additional retry needed since `enqueueWithRetry` already retried internally. An OPTIONAL second tryReclaim+`enqueueWithRetry` can be added for belt-and-suspenders, but is NOT necessary.
- **Category B (ISRRetireRouter sites):** Capture `RetireEnqueueResult`. On `!= Success`: log + `deleter(ptr)`. Since `ISRRetireRouter::enqueueWithRetry` already does 2 internal retries, NO additional retry is needed or recommended.
- **Unified helper (proposed):** Replace all inline recovery blocks with:
  ```cpp
  // For SnapshotCoordinator sites (Category A):
  static void retireOrFail(convo::IEpochProvider& provider, void* ptr,
                           void (*deleter)(void*), uint64_t epoch,
                           const char* debugLabel) noexcept {
      if (enqueueWithRetry(provider, ptr, deleter, epoch)) return;
      juce::Logger::writeToLog("SnapshotCoordinator: " + juce::String(debugLabel) + " — direct delete (NonRT)");
      deleter(ptr);
  }
  // For ISRRetireRouter sites (Category B):
  // (ISRRetireRouter::retire already wraps the call — just add deleter(ptr) in the failure branch)
  ```

**Thread Safety Analysis:**
- `SnapshotCoordinator.cpp:38` (`startFade`) — called from `AudioEngine.Timer.cpp` (Message Thread) → NonRT ✅
- `SnapshotCoordinator.cpp:94` (`completeFade`) — called from `AudioEngine.Timer.cpp:896` (Message Thread) → NonRT ✅
- `SnapshotCoordinator.h:88,100` (`switchImmediate`) — called from `AudioEngine.Timer.cpp` (Message Thread) → NonRT ✅
- `SnapshotCoordinator.h:158,160` (`retireCurrentAndTarget`) — called from `AudioEngine.Timer.cpp` (Message Thread) → NonRT ✅
- `ISRRetireRouter.cpp:154` (`retire`) — called from `onPublishCompleted` (worker thread, not audio callback) → NonRT ✅
- `DSPLifetimeManager.cpp:49,90` — called during cleanup/destruction (Message Thread) → NonRT ✅

All callers are NonRT — direct `delete` is safe at all sites.

**Memory Ordering:** No explicit memory ordering needed for the recovery path — `enqueueRetire` and `deleter` use the RCU epoch barrier internally. The atomic pointer swap (for BUG-014) is the only path requiring acquire/release ordering.

**Logging Protocol:** Each failure path logs exactly once before direct delete. Log messages include the function/site name and "(NonRT)" indicator for diagnostics.

**Testing Approach:**
- Inject a mock `IEpochProvider` that returns `false` from `enqueueRetire` to simulate queue pressure
- Verify `deleter(ptr)` (direct delete) is called exactly once on failure
- Assert no double-free: the pointer is only deleted once (either via `enqueueRetire` or `deleter`, not both)
- For Category B: inject `ISRRetireRouter` that returns `RetireEnqueueResult::QueuePressure` — verify direct delete is called

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

### BUG-028 Fix (step-by-step)

**`start()` — after `fadeStartTimestampUs_` publish (h:49):**
```cpp
convo::publishAtomic(firstIrDryDone_, false, std::memory_order_release);
convo::publishAtomic(dryScaleTarget_, 1.0, std::memory_order_release);
dryScaleGain_.setCurrentAndTargetValue(1.0);
```

**`complete()` — after `fadeStartTimestampUs_` publish (h:102):**
```cpp
convo::publishAtomic(dryScaleTarget_, 1.0, std::memory_order_release);
convo::publishAtomic(startDelayBlocks_, 0, std::memory_order_release);
convo::publishAtomic(dryHoldSamples_, 0, std::memory_order_release);
dryScaleGain_.setCurrentAndTargetValue(1.0);
```

#### Detailed Design — BUG-028

**Design Rationale:** `CrossfadeRuntime` maintains several fields that control the dry signal mixing path. `reset()` resets all fields but is only called once at construction. `start()` and `complete()` — called at the beginning and end of each fade cycle — do not reset the dry-scale fields, causing stale values from a previous fade to persist into the next cycle.

**Fields Requiring Reset in `start()`:**
| Field | Type | Old Value (from prior cycle) | New Value | Why |
|-------|------|------------------------------|-----------|-----|
| `firstIrDryDone_` | `bool` atomic | `true` (previous cycle completed) | `false` | Ensures IR dry path is re-evaluated for first input |
| `dryScaleTarget_` | `double` atomic | Stale gain target | `1.0` | Default: dry signal at full volume |
| `dryScaleGain_` | `LinearRamp` | Stale interpolated value | `1.0` | Reset both current and target to 1.0 |

**Fields Requiring Reset in `complete()`:**
| Field | Type | Old Value | New Value | Why |
|-------|------|-----------|-----------|-----|
| `dryScaleTarget_` | `double` atomic | Fade-end target | `1.0` | Post-fade: dry at full volume |
| `startDelayBlocks_` | `int` atomic | Delay applied during fade | `0` | No delay after fade completion |
| `dryHoldSamples_` | `size_t` atomic | Samples held during IR loading | `0` | No hold after completion |
| `dryScaleGain_` | `LinearRamp` | Faded gain value | `1.0` | Reset ramp for next cycle |

**RT-Safety:** All callers of `start()` and `complete()` are NonRT:
- `start()`: Called from `AudioEngine.Timer.cpp` (Message Thread) and `DSPTransition.h:66` (emergency override, NonRT)
- `complete()`: Called from `AudioEngine.Timer.cpp:896,1580` (Message Thread) and `DSPTransition.h:66,126` (NonRT)

`dryScaleGain_.setCurrentAndTargetValue(1.0)` is a synchronous ramp reset — safe on NonRT thread.

**Memory Ordering:** `convo::publishAtomic` uses `memory_order_release` — appropriate because the Audio Thread reads these via `convo::consumeAtomic` (acquire) before the fade cycle begins. The `start()`/`complete()` writes are sequenced before the audio thread's first read in the next cycle.

**Interaction with Existing Workaround:** The Timer path at `AudioEngine.Timer.cpp:896-898` already manually calls `setStartDelayBlocks(0)` + `setDryHoldSamples(0)` after `complete()`. Once `complete()` internally resets these fields, the manual workaround becomes redundant. However, the second `complete()` call at `AudioEngine.Timer.cpp:1580` only calls `setDryHoldSamples(0)` (line 1583) — `startDelayBlocks_` is NOT reset there. The fix in `complete()` will cover both Timer paths. The manual workaround lines can be removed after verifying the fix, but should be kept as defensive redundancy during transition.

**Testing Approach:**
- Unit test: call `start()` then immediately check all fields via `getXxx()` — verify `firstIrDryDone_==false`, `dryScaleTarget_==1.0`, `dryScaleGain_.getCurrentValue()==1.0`
- Unit test: call `complete()` then check all fields — verify `dryScaleTarget_==1.0`, `startDelayBlocks_==0`, `dryHoldSamples_==0`, `dryScaleGain_.getCurrentValue()==1.0`
- Integration test: trigger multiple fade cycles (start → complete → start → complete) and verify dry scale gain is 1.0 at the start of each cycle
- Regression test: verify Timer path behavior unchanged after removing manual workaround (if removed)

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
2. [ ] BUG-015: Category A (SnapshotCoordinator::enqueueWithRetry, bool): capture return at h:100, h:158, h:160; replace TODO at cpp:38, cpp:94 (also BUG-027), h:88 with direct delete. Category B (ISRRetireRouter::enqueueWithRetry, RetireEnqueueResult): replace TODO at ISRRetireRouter.cpp:154; remove `ignoreUnused` at DSPLifetimeManager.cpp:49,90 and add direct delete (NO extra retry — already done internally)
3. [ ] BUG-027: Apply Category A recovery to `completeFade` at `SnapshotCoordinator.cpp:94` (same as BUG-015 completeFade site — already listed in #2)
4. [ ] BUG-028: Add missing resets to `CrossfadeRuntime::start()` and `complete()` in `src/audioengine/CrossfadeRuntime.h`
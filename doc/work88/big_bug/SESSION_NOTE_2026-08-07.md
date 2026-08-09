# Session Note — 2026-08-07

## Comprehensive verification of INTEGRATED_BUG_LIST.md completed

### Phase 1: Initial verification & corrections
- Verified all 30 bugs against source code
- Corrected Bug 3-6 (NaN analysis was inverted)
- Fixed line numbers in Bugs 1-5, 1-7, 1-9
- Added investigation notes (Bug 1-3: additional unsafe sites, Bug 1-4: DefaultFastTanhPolicy, Bug 1-8: MAX_FILE_LENGTH guard)
- Added detailed fix designs for all 31 bugs in Section 7

### Phase 2: Unchecked mini-bugs verification (NEW)
- Verified 22 unchecked mini-bugs from `doc/work88/mini_bugs_unchecked/` against current source code
- **21 bugs verified as Fixed** in the current codebase
- **1 bug** (BUG-034) overlaps with Bug 1-6 (already documented)

### New R-entries added (R-17 through R-38):
| R# | Original BUG | Status | Fix location |
|----|-----|--------|-------------|
| R-17 | BUG-039 Oversampler buffer overread | ✅ Fixed | CustomInputOversampler.cpp:840-842 |
| R-18 | BUG-016 CmaEs sanitize NaN/Inf | ✅ Fixed | CmaEsOptimizer.h:208, CmaEsOptimizerDynamic.h:50 |
| R-19 | BUG-015 enqueueWithRetry return ignored | ✅ Fixed | ISRRetireRouter.cpp:154, SnapshotCoordinator.cpp:57,114 |
| R-20 | BUG-029 DSPTransition Emergency Override | ✅ Fixed | DSPTransition.h:63-78 |
| R-21 | BUG-014 currentDeviceTypeName_ CoW | ✅ Fixed | Refactored to enum atomic publish (AudioEngine.h:2354-2366) |
| R-22 | BUG-038 SpectrumAnalyzer +6 dB | ✅ Already correct | (Bug report was outdated) |
| R-23 | BUG-036 irL/irR release leak | ✅ Fixed | LoadPipeline.cpp:640-648 |
| R-24 | BUG-035 isLoading stuck | ✅ Fixed | ApplyComputedIRLoadingGuard RAII (LoadPipeline.cpp:325-338) |
| R-25 | BUG-019 TPD int overflow | ✅ Fixed | TruePeakDetector.cpp:102-103 (size_t) |
| R-26 | BUG-018 FP != 1.0 | ✅ Fixed | (All sites eliminated) |
| R-27 | BUG-021 timer no RCU guard | ✅ Fixed | Lifecycle.cpp:144-151 |
| R-28 | BUG-022 prepareToPlay no RCU guard | ✅ Fixed | Lifecycle.cpp:211-217 |
| R-29 | BUG-031 updateAudioThreadSnapshotFade stub | ✅ Fixed | DELETED (AudioEngine.h:3880) |
| R-30 | BUG-033 BlockDouble dryScale | ✅ Fixed | BlockDouble.cpp:420-427 |
| R-31 | BUG-042 CmaEs Rule of Five | ✅ Fixed | CmaEsOptimizer.h:43-46 |
| R-32 | BUG-045 IRConverter resample mislabel | ✅ Fixed | IRConverter.cpp:269-270 |
| R-33 | BUG-028 CrossfadeRuntime stale flags | ✅ Fixed | CrossfadeRuntime.h:106-110 |
| R-34 | BUG-041 NSL VLA stack overflow | ✅ Fixed | (replaced with vector/heap) |
| R-35 | BUG-034 IPP FFT MKLNonUniformConvolver | ⚠️ Bug 1-6 | (Overlaps with existing) |
| R-36 | BUG-024 SnapshotFadeState race | ✅ Fixed | SnapshotFadeState.h:67-73 |
| R-37 | BUG-037 loaderTrashBin UAF | ✅ Fixed | StateAndUI.cpp:977-987 |
| R-38 | BUG-046 PsychoacousticDither Rule of Five | ✅ Fixed | PsychoacousticDither.h:98-105 |

### Summary statistics:
- INTEGRATED_BUG_LIST.md: **1101 lines** (was 1079)
- **38 R-entries** (was 16): 21 ✅ Fixed, 14 ❌ Rejected (R-1 to R-8, R-10 to R-15), 2 ✅ Confirmed (R-9, R-16), 1 ⚠️ (R-35)
- **33 ✅ Confirmed** active bugs
- **21 newly verified Fixed** (from unchecked mini-bugs)

### Files modified:
- doc/work88/big_bug/INTEGRATED_BUG_LIST.md (1101 lines)

## Next steps:
- Apply open P0 fixes (Bugs 1-1, 1-3, 1-6, 1-7, 1-8, 1-9)
- Fix CMakeLists.txt: Remove /fp:fast and /QxCORE-AVX2
- Verify remaining unchecked bugs (BUG-020, BUG-025, BUG-026, BUG-027, BUG-030, BUG-032, BUG-040, BUG-043, BUG-044)

# Session Note — 2026-08-07

## Comprehensive verification of INTEGRATED_BUG_LIST.md completed

### Corrections made (verified against source code):
1. **Bug 3-6 (NaN hash)**: Analysis was INVERTED — NaN causes FALSE POSITIVE (NaN treated as equivalent), not false negative. Fixed phenomenon text.
2. **Bug 1-7 (ISRRetire mutex)**: Updated file reference from h:136 → h:169 (actual location of fallbackMutex_)
3. **Bug 1-5 (musicalSoftClip)**: Updated line ref from h:1059 → h:1066. Confirmed 0 callers of class method — only file-local musicalSoftClipScalar is used.
4. **Bug 1-4 (fastTanh dup)**: Added note about DefaultFastTanhPolicy at FastTanhApprox.h:28, used by EQProcessor.Processing.cpp:104. Duplicates should use SoftClipPadéPolicy (same as DSPCoreDouble).
5. **Bug 1-3 (_mm256_store_pd)**: Added note about 3 additional unsafe sites: TruePeakDetector.cpp:85, EQProcessor.Processing.cpp:37, MKLNonUniformConvolver.cpp:1319/1580
6. **Bug 1-8 (OOM)**: Added note about MAX_FILE_LENGTH guard at LoaderThread.cpp:450, ResampleAndFallback.cpp:293 (prevents integer overflow but not memory exhaustion)
7. **Bug 1-9 (int/size_t)**: Updated line numbers to 843, 847, 853. Updated fix: static_cast<size_t> alone insufficient — fftSize itself should be int64_t
8. **Bug 1-9/1-10 ordering**: Moved Bug 1-10 after Bug 1-9 for proper numbering

### Previously corrected (from prior session):
- Bug 2-10: NoiseShaperType enum values (Adaptive9thOrder=2, Fixed15Tap=3, not swapped)
- Bug 2-7: static_assert IS active for non-MSVC compilers (h:186)
- Bug R-9: CMakeLists.txt DOES use set(CMAKE_CXX_FLAGS_RELEASE) — Confirmed
- Bug R-16: /fp:fast confirmed as separate entry
- Bug 1-10, 3-9, 3-10: 3 new bugs added

### Detailed fix designs added:
- New section 7: "詳細修正設計 (Detailed Fix Design)" with designs for ALL 30 bugs + R-9
- Each design includes: Root Cause, Fix Approach (with code), Testing, Risk
- Section 8: "未調査領域" updated with investigation status for each area

### Source verification method:
Used `src/` grep tool for code verification, WSL bash (rg, sed) for line number checks. Verified:
- getState/setState: nucHCMode/nucLCMode confirmed absent from persistence
- coordinatorDeferredRing_: confirmed no producer pushes (only pop/decommit)
- musicalSoftClip: confirmed 0 callers (class method is dead code)
- MklFftEvaluator.h:270-271,425-426: confirmed IPP return values unchecked
- ISRRetire.h:169: confirmed fallbackMutex_ used in RT path
- MAX_FILE_LENGTH guard: confirmed present (limits to INT32_MAX, not memory-safe)

### Files modified:
- doc/work88/big_bug/INTEGRATED_BUG_LIST.md (1079 lines)

## Next steps:
- Apply the 4 open fixes from REPAIR_PLAN3.md
- Implement Bug 1-1: Add nucHCMode/nucLCMode to getState/setState
- Implement Bug 1-10: Delay m_pendingIRChange clear until after publish
- Fix CMakeLists.txt: Remove /fp:fast and /QxCORE-AVX2 from global flags
- Rename SoftClipPadéPolicy → SoftClipPadeApproxPolicy (Bug 2-1/1-4)

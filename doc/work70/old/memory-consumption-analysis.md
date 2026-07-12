# Memory Consumption Analysis Report

**Log**: ConvoPeq.log (30,457 lines, ~163.5s runtime)
**Scenario**: Silent startup → IR file load + PEQ config → Adaptive NoiseShaper Continuous mode → Music playback → Shutdown
**Date**: 11 Jul 2026

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Total runtime | 163.5 seconds |
| Peak Private memory | 1,094 MB (gen=7 rebuild) |
| Steady-state Private memory | 683–686 MB |
| Baseline delta (start→end steady) | +38 MB (648→686 MB) |
| Memory leaks detected | **No leak observed in 163s window** (caution: short window) |
| Unnecessary allocations | 1 (obsoleted gen=1 DSPCore at 768k rate) |
| CPU waste (non-memory) | NoiseShaper: all 3000+ samples/sec dropped (sample rate mismatch) |

## 2. Full Memory Timeline

```
Time(s)  Event                              Private    Delta   Notes
-------  ---------------------------------  ---------  ------  ---------------
  0.0    Log started                        75 MB      —      JUCE framework + EQ ctor
  0.4    gen=1 build (48kHz, 384k rate)     556 MB     +481   First DSPCore alloc
  0.6    prepareToPlay(192kHz)              556 MB     0      gen counter reset
  0.8    gen=3 build (192kHz, osFactor=2)   996 MB     +440   Two DSPCores coexist
  1.0    gen=3 published                    648 MB     -348   Old DSPCore released
  1.5    Heap warmup complete               674 MB     +26    Buffer commit + C++ heap
+1.1–5.9 Steady state #1                   674→683 MB  +9     Stable audio processing

 +5.9    rebuild gen=4/5/6/7 start          —          —      convolverParamsChanged
 +5.9    gen=7 build (IR loaded)            1063 MB    +380   BUILD_PHASE memBuild
 +5.9    gen=7 build peak                   1094 MB    +31    IR + MKL workspace
 +5.9    AUTH_CONTRACT FAIL                 —          —      fadingNode=0 hasFading=1
 +5.9    IR release cleanup                 802→676 MB -418   MKL persistent → 0MB
 +6.0    Steady state #2 resumes            676→686 MB  +10   Same baseline as #1

+10.9    NoiseShaper learning starts        684 MB     0      accepted=0 (all dropped)
+12.5    —                                   686 MB     0      Stable
+15.8    —                                   686 MB     0      Stable
+18.7    —                                   686 MB     0      Stable
+20.0    —                                   686 MB     0      Stable
+163.5   Shutdown                            —          —      Clean release

Memory Peaks (rebuild sub-cycle):
  rec=104  890 MB    — gen=4 prepare (EQ buffers)
  rec=106  712 MB    — partial cleanup
  rec=108  991 MB    — gen=5 prepare (DSPCore alloc)
  rec=110  954 MB    —
  rec=112  1047 MB   — gen=6 prepare
  rec=114  1063 MB   — gen=7 build (build phase)
  rec=116  1074 MB   — gen=7 (IR load in progress)
  rec=118  1094 MB   — **PEAK**: gen=7 + MKL workspace
  rec=120  802 MB    — IR released, NUC freed
  rec=122  676 MB    — Back to baseline
```

## 3. Memory Composition Breakdown

### 3.1 Steady-State Composition (683–686 MB)

The `MEM_SNAP` reports:
- **NUC**: live=0, alloc=0MB, peak=35MB — No convolution IR loaded in steady state
- **DC**: live=1 — One DSPCore instance active
- **SC**: live=0 — No shared convolution objects
- **Ret**: pend=0, tr=0/0, reclaim=0 — No EBR activity
- **Other** = 683–686 MB — **All remaining memory, undifferentiated**

**CRITICAL CAVEAT**: The `Other` field is a black box in the current diagnostic output.
**It cannot be attributed to any specific component.** The following is a list of ALL
components that reside within this 683–686 MB, not an attribution:

Components that coexist within `Other` (no breakdown available), organized by category:

**Application-owned**:
- DSPCore with all sub-components (EQProcessor, Convolver, Oversampler, Limiter,
  Meter, NoiseShaper, SoftClipper, DC blockers, output filter, peak limiter,
  loudness meter, true peak detector)
- RuntimeWorld (active generation's runtime state)
- RuntimeBuilder temporary buffers (during rebuild)
- Transition/crossfade state
- Snapshot coordinator state
- IO buffers (input/output ring buffers)
- Capture buffers
- Telemetry/buffers

**Allocator-owned (C++ runtime + OS)**:
- C++ CRT heap (fragmentation, free list overhead, reserve vs commit)
- VirtualAlloc reservation / commit behavior
- Thread stacks (audio thread + worker threads + NoiseShaper thread)
- DLL/static library image mappings

**External library**:
- MKL convolution internal state (kernel memory, descriptors, scratch buffers)
- FFT plan cache
- IPP internal state

**Framework**:
- JUCE framework (AudioProcessorGraph, component tree, graphics resources):
  typically **tens of MB, not hundreds** — JUCE alone cannot explain 680 MB

**Unclassified**:
- OS memory manager overhead

**Key conclusion**: The 683+ MB baseline cannot be attributed to any single component
with the current diagnostics. A memory-bucket DIAG tool is required to partition this.
JUCE is NOT "dominant" — it is typically tens of MB. MKL is NOT hundreds of MB either,
especially with `MKL_THREADING=sequential`. The 680 MB is the aggregate of many
components plus possible allocator overhead / fragmentation candidate and VirtualAlloc reservation/commit patterns.

### 3.2 Allocations During Rebuild Cycle

| Buffer | Size | When |
|--------|------|------|
| DSPCore internalMaxBlock=524288 | ~0.5 MB | Each prepare |
| EQ scratch (processingRate×2ch) | 4 MB | Each prepare |
| EQ msWorkBuffer | 2 MB | Each prepare |
| NUC IR persistent data (2ch × 18MB) | 36 MB | IR load |
| NUC MKL workspace (peak before→after delta) | 35→17 MB | IR load |
| NUC MKL persistent (per NUC) | 17 MB | After IR load |

### 3.3 Peak Memory Contributors

**996 MB at gen=3 build**: Two DSPCore instances coexist:
1. Old gen=1 DSPCore (48kHz, processingRate=384k, blockSize=524288) — not yet released
2. New gen=3 DSPCore (192kHz, processingRate=384k, blockSize=2048) — being prepared
→ Delta from baseline: **~350 MB increase**

**Updated understanding (code investigation 2026-07-11)**:
The ~350 MB delta BREAKS DOWN as:
- **Tracked buffer allocations**: ~159 MB (directly computed from code formulas)
  - Oversampling work buffers: 64 MB (2ch, 4x OS, internalMaxBlock×4×8 each)
  - SoftClip OS work buffers: 32 MB (2ch, 2x OS, internalMaxBlock×2×8 each)
  - EQ Processor: 21 MB (scratch/dry/parallel/structure/msWorkBuffer)
  - Aligned + dryBypass: 16 MB (L+R, internalMaxBlock×8 each)
  - Latency buffers: 16 MB (4× oldL/oldR/newL/newR)
  - TruePeakDetector: 8 MB
  - Crossfade: 1.5 MB
  - Others (Convolver/LoudnessMeter/DCBlocker/PeakLimiter/NoiseShaper): <1 MB
- **Hidden overhead**: ~191 MB — **unmeasured residual. Candidate contributors**
  include: possible allocator overhead / fragmentation candidate, VirtualAlloc granularity, CRT metadata,
  MKL FFT plan internal workspace, std::vector/string internals. This is a
  **list of candidates, not a confirmed breakdown**.

**Key insight**: `internalMaxBlock=524288` is IDENTICAL for gen=1 (spb=65536) and gen=3
(spb=1024) because `inputMaxBlock = max(SAFE_MAX_BLOCK_SIZE=65536, spb)` always returns
65536. DSPCore buffer allocations are the SAME regardless of samplesPerBlock.
The ~481 MB gen=1 delta vs ~350 MB gen=3 delta difference includes a one-time
initialization cost (~130 MB) that encompasses JUCE framework init, CRT startup,
MKL/IPP one-time allocations, FFT plan construction, and VirtualAlloc heap
reservation expansion — not attributable to any single component.

**1,094 MB at gen=7 rebuild**: Triple allocation:
1. Active runtime baseline (DSPCore + RuntimeWorld + heap + framework combined):
   ≈650 MB (very rough upper bound: 686 MB baseline minus ~36 MB directly attributable
   to NUC+SC — DSPCore alone is a subset of this)
2. Build DSPCore (gen=7): ~350 MB (same breakdown as above: 159 MB tracked + 191 MB hidden)
3. NUC IR data + MKL workspace: ~80 MB (directly measured from IR_LAYOUT logs: 36 MB
   persistent + ~44 MB transient MKL workspace)

→ Peak alloc = baseline + rebuild + convolution

## 4. Memory Leak Detection Results

### 4.1 MKL Convolution Layer
```
[IR_RELEASE] NUC#...FE4E40: before=35MB after=17MB delta=-17MB lostFree=0
[IR_RELEASE] NUC#...F44C00: before=17MB after=0MB delta=-17MB lostFree=0
```
**lostFree=0 on all releases** → No MKL memory leaks ✅

### 4.2 NUC Lifecycle
- NUC alloc=0MB in steady state (no IR loaded after AUTH_FAIL)
- NUC peak=35MB during gen=7 rebuild (fully recovered)
- No orphaned NUC count (live=0)

### 4.3 EBR Retirement

```
Ret: pend=0 trBytes=0.0MB tr=0/0(0%) ovf=0
```
- No pending retirements
- No transfer bytes
- Zero retirement attempts

**⚠️ Important caveat**: `retire=0` does NOT automatically mean "no leak".
In this log, the P1-a/FIX implementation ensures DSPCore handles are properly
registered. The `reclaim=0` simply means EBR reclamation was never triggered because:
- No Reader (audio callback) reference counting was active during the observed window
- The single active DC was never replaced (gen=3 published once, no subsequent
  successful publish due to AUTH_CONTRACT FAIL)

Correct interpretation: **EBR was not exercised** during this log window, so
no EBR-related leak conclusions can be drawn. A proper EBR exercise requires a
successful Publish → Observe → Retire sequence with active Reader references.

### 4.4 Transaction Counters
```
tx counters lifecycle(pub/ret/reclaim)=4/0/0
```
- `reclaim=0` throughout: EBR reclaimer never activates (no opportunity to verify)

### 4.5 Heap Growth Over 163 Seconds

Steady-state memory is confined to **683–686 MB** for the final ~160 seconds.
No detectable progressive growth pattern → **No heap leak detected in this window** ✅

The +38 MB from 648→686 MB in the first ~5 seconds is normal heap warmup:
- C++ runtime heap expansion
- VirtualAlloc lazy commit
- Audio processing buffer commit on first touch

**Caution**: 163 seconds is a short observation window. Long-running scenarios
(8+ hours, 500+ IR switch cycles) may reveal different behavior.

## 5. Unnecessary Allocation Findings

### 5.1 [MINOR] Obsoleted gen=1 DSPCore at 768k Processing Rate

**Observation**: During gen=2→3 transition, a DSPCore prepare ran at
`processingRate=768000 processingBlockSize=4096` (line 126). This was immediately
obsoleted by gen=3 which selected `processingRate=384000 processingBlockSize=2048`.

**Root Cause**: The rebuild-request cascade (gen=1→gen=2→gen=3) caused gen=1's DSPCore
prepare to run with different oversampling parameters before gen=3 obsoleted it.

**Impact**: ~100ms wasted CPU, memory was freed when gen=3 published. No permanent leak.
**Severity**: Cosmetic. Architectural double-buffering requires this.

### 5.2 [INFO] 648→686 MB Initial Growth

The +38 MB growth in the first ~5 seconds is attributed to:
- Lazy page commit of reserved virtual memory (C++ `new` often reserves more than it commits)
- JUCE audio buffer ring on first audio data flow
- Normal heap manager expansion

**Verdict**: No evidence of a real leak. All growth occurs in the first ~5 seconds and
then plateaus perfectly flat for 160+ seconds.

### 5.3 [NON-MEMORY] NoiseShaper Learner Sample Rate Mismatch

The NoiseShaper learner (started at +10.9s) shows `accepted=0` on ALL diagnostic samples.
Every ~730 lines, the learner reports `dropSampleRate=~3000` samples dropped.

**Root cause (code investigation 2026-07-11 — UPDATED)**:
- Session sample rate: `engine.currentSampleRate` (192000 Hz for gen=3)
  → `captureSessionSignature()` at `NoiseShaperLearner.cpp:1034`
- Capture block sample rate: `dsp->sampleRate` (192000 = base rate, NOT processing rate)
  → `buildAudioThreadProcessingState()` at `AudioEngine.h:3552` uses `static_cast<int>(dsp->sampleRate + 0.5)`
  → This is passed as `state.adaptiveCaptureSampleRateHz` to `pushAdaptiveCaptureBlocks()`
  → `DSPCoreDouble.cpp:291`: `block.sampleRateHz = sampleRateHz`
- `dsp->sampleRate` is set to `newSampleRate` in `DSPCoreLifecycle.cpp:104` (the base sample rate)
- Therefore `block.sampleRateHz == session.sampleRateHz (both 192000)` should hold.

**The earlier conclusion ("processing rate bug") was INCORRECT.**
The rate values are correctly set from the base sample rate. Additional code investigation (2026-07-11) confirmed:
- `currentCaptureSessionId` at `AudioEngine.h:943` is **initialized to 0 and NEVER written** — no code path modifies it
- Both `block.sampleRateHz` and `session.sampleRateHz` derive from the same `dsp->sampleRate` (base rate)
- The shared 4096-entry ring buffer is fully overwritten by gen=3's ~10s of audio before NoiseShaper learner starts, ruling out stale gen=1 blocks

The root cause of `droppedBySampleRate>0` remains unknown and **cannot be resolved by code analysis**:
- Requires RUNTIME investigation: add DIAG output of actual block.sampleRateHz and session.sampleRateHz values
  used in the drop decision, then observe at runtime
- Undiagnosed segmentBuffer push failure path

**Impact**: NoiseShaper learner is non-functional, estimated ~240K samples CPU wasted per loop.
**Severity**: Functional bug, needs runtime diagnosis. Not a memory issue.
**Status**: 🔍 Changed from FACT #73 to **HYPOTHESIS** — root cause requires runtime observation.

## 6. AUTH_CONTRACT FAIL Analysis

```
[AUTH_CONTRACT] FAIL fadingNode=0 hasFading=1
```

The gen=7 rebuild (with IR loaded) was rejected by the Runtime Graph Authority because:
- `fadingNode=0`: No FadingNode exists in the runtime graph
- `hasFading=1`: But the fading flag is set (crossfade in progress from previous change)

**Impact**: IR data (36 MB persistent + 35 MB MKL workspace) was allocated, prepared, then
immediately released. This caused the 1,094 MB peak. After cleanup, no IR is active.

This is a pre-existing authority contract issue (already documented in modification-plan-v3.md).

## 7. Recommendations

### 7.1 Diagnostic Improvements
- **Add memory bucket tags to MEM_SNAP** to identify the ~680 MB "Other" composition
  (at minimum: DSPCore, RuntimeWorld, MKL, JUCE, CRT heap, VirtualAlloc reservation)
- **Add RuntimeWorld per-generation memory tracking**: track each gen's contribution
- **Add VirtualAlloc Reserve vs Commit breakdown**: distinguish reserved virtual address
  space from actually committed physical memory in Private Bytes
- **Add MKL/IPP allocator tracking**: instrument mkl_malloc/mkl_free that escape the
  DSPCore lifecycle (CacheManager, IRConverter)
- **Add RuntimeBuilder temporary buffer total**: count temporary allocations during build
- **Add Retire-queue object count + total memory**: for when EBR is exercised
- Add NoiseShaper `expectedSampleRate vs actual` diagnostic to identify mismatch
- Add rebuild-phase annotations to MEM_SNAP (which generation is preparing)

### 7.2 Optimization Candidates
- Investigate 680+ MB baseline with proper bucket DIAG before concluding any single
  component is responsible — JUCE alone cannot explain 680 MB (typically tens of MB)
- Consider lazy commit for large EQ buffers only when nonzero signal detected
- Fix NoiseShaper sample rate detection to avoid useless CPU burn

### 7.3 Fix Required
- AUTH_CONTRACT FAIL at gen=7 prevents IR application (pre-existing issue)

## 8. Conclusion

**No memory leak was observed within the 163-second window** — the 648–686 MB steady
state is stable for 160+ seconds with zero detectable growth. The 1,094 MB peak during
IR rebuild is architecturally expected (double-buffering + convolution data).

However, the following limitations apply:
1. **Short observation window (163s)**: Long-running or high-iteration scenarios may
   reveal different behavior.
2. **Undifferentiated 680 MB "Other"**: The single largest uncertainty. Current MEM_SNAP
   cannot attribute this to specific components. JUCE+MKL attribution is speculative.
3. **EBR not exercised**: The retire path was never activated during this log window,
   so EBR correctness cannot be verified from this data.
4. **DSPCore cost is derivative, not direct**: The ~350 MB per DSPCore figure is a
   subtraction-based estimate, not a direct measurement.

The primary value of this analysis is the timeline, peak identification, AUTH_CONTRACT
FAIL impact assessment, and the observation that DC live=1 (single DSPCore) was
maintained throughout this log, in contrast to the previously observed DC live=3
which indicated rebuild-obsolete DSPCore retention.

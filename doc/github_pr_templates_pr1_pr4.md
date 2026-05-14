# ConvoPeq PR Templates (PR1-PR4)

このファイルは、実装順序を固定した PR1〜PR4 の GitHub PR テンプレートです。
各セクションの Title をPRタイトルに、Body をPR本文にそのまま貼り付けて使用します。

---

## PR1

### Title

[Runtime Invariants] PR1: Remove published-runtime mutation and prepare crossfade on Message Thread

### Body

## Summary

This PR restores core runtime immutability by removing direct mutation of published DSP runtime objects and starting crossfade-prepare migration to the Message Thread.

## Scope

- Remove direct published runtime mutation from convolver UI event flow
- Keep convolver change handling as diff-detection + rebuild request only
- Introduce/prepare crossfade prepared-state pathway on commit side

## Files

- src/audioengine/AudioEngine.UIEvents.cpp
- src/audioengine/AudioEngine.Commit.cpp
- src/audioengine/AudioEngine.h

## Why

- IR-1 Render-phase immutability
- IR-3 Ownership-fixed state changes
- IR-6 Ownership transfer discipline

## Implementation Checklist

- [ ] Remove activeDSP->convolverRt().syncParametersFrom(...) call path
- [ ] Ensure convolverParamsChanged performs only diff detection and requestRebuild
- [ ] Add crossfade prepared-state data model (or equivalent)
- [ ] Compute fadeSec/delayOld/delayNew/useDryAsOld/startDelayBlocks on commit side
- [ ] Publish prepared transition state for Audio Thread consumption

## Validation Checklist

- [ ] grep confirms no activeDSP->convolverRt().syncParametersFrom usage
- [ ] get_errors reports no new diagnostics in changed files
- [ ] Rebuild still triggers correctly for convolver structural changes
- [ ] IR load/reload path has no crash or silent regression

## Risks / Notes

- Structural rebuild dedup logic must remain intact
- No behavior changes in audio path are intended in this PR

---

## PR2

### Title

[RT Safety] PR2: Make Audio Thread crossfade path consume-only and introduce publication helpers

### Body

## Summary

This PR completes crossfade responsibility split by removing ramp initialization from the Audio Thread and introducing atomic publication helpers for runtime transition state.

## Scope

- Convert Audio Thread crossfade path to activate-only behavior
- Remove reset/initialization from Audio Thread crossfade entry
- Introduce helper-based publication API for key runtime pointers

## Files

- src/audioengine/AudioEngine.h
- src/audioengine/AudioEngine.Processing.BlockDouble.cpp
- src/audioengine/AudioEngine.Processing.AudioBlock.cpp

## Why

- IR-1 Render-phase immutability
- IR-2 Single publication world
- IR-4 No hidden synchronization/initialization on RT path

## Implementation Checklist

- [ ] Remove dspCrossfadeGain.reset(...) from Audio Thread path
- [ ] Remove setCurrentAndTargetValue(0.0) from Audio Thread path
- [ ] Keep Audio Thread logic to consume prepared state and activate once
- [ ] Add helper APIs (publish/consume/exchange atomic pointer patterns)
- [ ] Replace direct atomic operations for current/fading transition points where in scope

## Validation Checklist

- [ ] grep confirms no crossfade ramp initialization in Audio Thread functions
- [ ] get_errors reports no new diagnostics in changed files
- [ ] Crossfade quality is preserved (no new click/pop on transitions)
- [ ] dry-as-old transition still works as designed

## Risks / Notes

- Ensure transition activation remains exactly-once
- Maintain existing latency-alignment behavior

---

## PR3

### Title

[EQ Runtime] PR3: Replace SmoothedValue with LinearRamp and lock down RCUReader ownership semantics

### Body

## Summary

This PR removes remaining SmoothedValue runtime dependence in EQ paths and formalizes RCUReader ownership constraints by prohibiting copy/move.

## Scope

- Replace EQProcessor smoothing members with convo::LinearRamp
- Align prepare/process logic with LinearRamp API semantics
- Delete copy/move operations for RCUReader

## Files

- src/eqprocessor/EQProcessor.h
- src/eqprocessor/EQProcessor.Core.cpp
- src/eqprocessor/EQProcessor.Processing.cpp
- src/core/RCUReader.h

## Why

- IR-4 RT-safe deterministic path constraints
- IR-5 Epoch-consistent observation discipline
- IR-6 Ownership transfer clarity

## Implementation Checklist

- [ ] Replace smoothTotalGain member type to convo::LinearRamp
- [ ] Replace bypassFadeGain member type to convo::LinearRamp
- [ ] Update prepare/process code to LinearRamp-compatible operations
- [ ] Preserve bypass transition completion semantics
- [ ] Add delete declarations for RCUReader copy/move ctor/assignment

## Validation Checklist

- [ ] grep confirms no juce::SmoothedValue runtime usage in EQProcessor
- [ ] grep confirms no forbidden libm additions on Audio Thread path
- [ ] get_errors reports no new diagnostics in changed files
- [ ] EQ on/off/bypass transitions remain stable under repeated interaction

## Risks / Notes

- Maintain current UX-level smoothing feel
- No topology changes should be introduced

---

## PR4

### Title

[Publication Semantics] PR4: Complete publication helper migration, coalesce commit notifications, and introduce aligned_make_unique

### Body

## Summary

This PR finalizes publication-edge consistency improvements, reduces commit-notification flooding via coalescing, and introduces aligned_make_unique as the standardized aligned allocation helper.

## Scope

- Expand helper-based publication semantics across remaining runtime edge points
- Add pending change-notification coalescing in commit notification path
- Introduce aligned_make_unique and start replacing eligible non-RT allocations

## Files

- src/audioengine/AudioEngine.h
- src/audioengine/AudioEngine.Commit.cpp
- src/eqprocessor/EQProcessor.Core.cpp (if allocation migration in scope)
- src/audioengine/DSPExecutionState.h (if allocation migration in scope)

## Why

- IR-2 Single publication world
- IR-5 Consistent epoch observation
- IR-7 Crossfade isolation and robust lifecycle boundaries

## Implementation Checklist

- [ ] Replace remaining key publication points with helper-based atomic semantics
- [ ] Ensure current/fading/runtimeGraph/engineRuntime publication order is consistent
- [ ] Add pendingChangeNotification (or equivalent) for commit notification coalescing
- [ ] Keep sendChangeMessage-trigger behavior functionally equivalent for UI expectations
- [ ] Add aligned_make_unique helper (64-byte alignment, exception-safe)
- [ ] Use helper for newly migrated non-RT allocation paths

## Validation Checklist

- [ ] grep confirms reduced direct memory_order usage in targeted publication boundaries
- [ ] get_errors reports no new diagnostics in changed files
- [ ] Repeated rebuild cycles do not produce notification storms
- [ ] ASan/Debug execution shows no new leak/UAF/double-free regressions

## Risks / Notes

- Keep migration incremental to avoid broad-scope regressions
- Maintain strict no-allocation/no-lock constraints on Audio Thread

---

## Shared Definition of Done (All PRs)

- [ ] No new compile/lint diagnostics in modified files
- [ ] No regression in build task execution (Debug or Release)
- [ ] No new RT-unsafe operations introduced in Audio Thread path
- [ ] PR body explicitly maps changes to IR-1..IR-7 invariants

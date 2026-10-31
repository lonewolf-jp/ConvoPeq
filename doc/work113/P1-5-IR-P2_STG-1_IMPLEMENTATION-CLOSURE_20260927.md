# STG-1 Implementation Closure

```text
Bug ID     STG-1 — Bypass Toggle Destroys User Staging Gains
Repair     Option 1 (remove 2 applyDefaultsForCurrentMode() calls)
Commit     6dc032133f365ed4ad6a525fe8353c54b7124d8f
Message    fix(audioengine): preserve staging gains across bypass toggles
Push       NOT AUTHORIZED (not executed)
```

## Pre-change gate (I5-R1-PRE equivalent)

```text
2 applyDefaultsForCurrentMode() call sites, both in bypass setters
6 symbols unaffected / 1 production file / 0 other files
mapping load/store/exchange/fetch_or: n/a (call removal, not substitution)
```

## Reproduction before repair

```text
[STG-1] FAIL: conv/makeup destroyed by bypass toggle: got 10.000000 want 3.000000
FAIL: checkSTG1BypassStagingPreservation
(harness exit 1, all prior checks PASS)
```

## Implementation (2 lines removed)

```text
src/audioengine/AudioEngine.Parameters.cpp
  setEqBypassRequested:        - applyDefaultsForCurrentMode();
  setConvolverBypassRequested: - applyDefaultsForCurrentMode();
applyDefaultsForCurrentMode() body: KEPT (0 callers; declaration kept)
setter contracts / bypass structure / all other logic: unchanged
```

## Regression tests (added, no new CTest entry)

```text
src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp
  checkSTG1BypassStagingPreservation (2 bypass x 3 staging = 6/6)
  checkSTG1StateIORoundTrip (save/toggle/save/load, 3/3)
```

## Validation

```text
post-repair harness : STG-1 6/6 PASS, StateIO 3/3 PASS, all prior checks PASS
Debug build         : PASS (gate OK)
Release build       : PASS (573/573)
CTest               : PASS (42/42, 100%)
```

## Authority

```text
previous : 6F1ED4815A4398FA3A5D34C6D72A914B63D9FDFC38E0BB2A6F2B47E743C90FB3
regenerated post-change, committed in 6dc03213:
  size      5,557,334 B
  SHA-256   3C61996501C6DA27AD85F36998E746301BC6CBF1BBED48F338C11015A508CD7A
  Generated 2026-09-27 23:12:13
sanity: bypass setters contain no applyDefaultsForCurrentMode() calls;
  sole remaining occurrence is the declaration; STG-1 test markers present
```

## ISR / architecture

```text
RT path / ISR Bridge / RuntimeWorld / Publication / Crossfade / Retire-Epoch /
Recovery-Rebuild ownership / atomic wrapper contract: all unchanged.
ASSERT_NON_RT_THREAD preserved. NonRT parameter path only.
```

## Excluded (pre-existing, untouched)

```text
M AGENTS.md / M headroom-proxy-start.ps1 — NOT in commit, NOT reverted.
```

## State

```text
STG-1 implementation = CLOSED (commit GO fulfilled)
files modified: 2 source/test + 1 authority artifact
subjects modified beyond scope: 0
```

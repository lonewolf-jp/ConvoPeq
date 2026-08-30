# D135-8 Step 8 — RebuildThread Ownership Assertion Audit (diff-only, no build/test)

**Status:** PASS · **Date:** 2026-08-30 · **Layer:** Implementation (P3 follow-on: ownership contract hardening)
**Files changed:** 1 — `src/audioengine/RuntimePublicationOrchestrator.h`
**Baseline:** HEAD `f39fcd3`. Audit is **read-only / diff-only** — no build, no CTest (per standing gate rule).

## Rationale
`resetDeferredRetryBudget()` mutates plain (non-atomic) members `deferredRetryGeneration_` / `deferredRetryCount_`, whose single owner is the RebuildThread. Step 5 removed the producer-side call (Timer.cpp); Step 7 relocated the **only** remaining call into `processDeferredAdmission(bool)` on the RebuildThread. This Step 8 makes that single-owner contract **explicit and fail-loud in debug builds** by adding the project's canonical owner assertion — the same `jassert` idiom already present in `processDeferredAdmission` (`cpp:637`) and two sibling functions (`cpp:565`, `cpp:590`).

## Change
`RuntimePublicationOrchestrator.h` — `resetDeferredRetryBudget()` definition (was `:171-174`, now `:171-175`):

```cpp
// ★ D135-1: recovery が zero-fading world を commit した時点で retry budget をリセット。
void resetDeferredRetryBudget() noexcept {
    jassert(std::this_thread::get_id() == engine_.rebuildThreadId());   // ← NEW: Step 8
    deferredRetryGeneration_ = 0;
    deferredRetryCount_ = 0;
}
```

The assertion idiom was **not** pasted blindly — it was verified against the latest `ConvoPeq.md` and source:
- Source (`rg -n 'jassert.*rebuildThreadId' src/audioengine/`): canonical form `jassert(std::this_thread::get_id() == engine_.rebuildThreadId());` at `cpp:565`, `cpp:590`, `cpp:637` (plus a comment at `AudioEngine.h:1663`).
- `ConvoPeq.md`: identical text at lines 66215 / 66240 / 66287.
- Indent matched to the body (8 spaces, same as the adjacent `deferredRetryGeneration_ = 0;` statement).

## 10-criterion read-only audit

| # | 確認 | 合格条件 | Result | Evidence |
|---|------|----------|--------|----------|
| 1 | assertion | `resetDeferredRetryBudget()` 冒頭にRebuildThread owner assertion | PASS | `h:172` `jassert(std::this_thread::get_id() == engine_.rebuildThreadId());` — matches canonical idiom |
| 2 | reset本体 | `deferredRetryGeneration_ = 0` / `deferredRetryCount_ = 0` 変更なし | PASS | `h:173`, `h:174` byte-identical to pre-Step-8 |
| 3 | caller | callerは `processDeferredAdmission(bool)` のみ | PASS | sole call `cpp:650` (inside `if (wasRecoveryWake)` :649) — see Step 7 |
| 4 | Timer.cpp | `resetDeferredRetryBudget()` の呼び出しなし | PASS | `rg` over `AudioEngine.Timer.cpp` → (none) |
| 5 | processDeferredAdmission | `if (wasRecoveryWake)` gatingを維持 | PASS | `cpp:649-650` intact |
| 6 | recovery provenance | `recoveryRetryReady` のproducer/consumer配線を変更しない | PASS | Step 8 edits only `resetDeferredRetryBudget` body; producer (`Timer.cpp:1748`) / consumer (`RebuildDispatch.cpp:888`) untouched |
| 7 | CV predicate | 変更なし | PASS | `RebuildDispatch.cpp:854-859` unchanged |
| 8 | changed files | `RuntimePublicationOrchestrator.h` のみ | PASS | 1 file edited; `.cpp`/`.Timer.cpp`/`.h` AudioEngine untouched by Step 8 |
| 9 | dead field | `deferredRecoveryRearmed_` が復活していない | PASS | `rg -n 'deferredRecoveryRearmed_' src/audioengine/` → (none — absent since Step 2) |
| 10 | retry cap | `kMaxDeferredRetries=2`、exhaustion `>` を維持 | PASS | spec comment `RuntimePublicationState.h:17` (`kMaxDeferredRetries=2`); exhaustion `if (deferredRetryCount_ > kMaxDeferredRetries)` `cpp:489` (Step 1) untouched |

## Diff stat (vs Step-7 on-disk state)
- `RuntimePublicationOrchestrator.h`: **+1 line** (`jassert(...)` inserted at start of `resetDeferredRetryBudget` body). Net surgical; no other file.

## Why this is safe (debug-only + existing pattern)
- `jassert` is a no-op in release/JUCE default builds (no runtime cost in production); in debug it hard-fails if `resetDeferredRetryBudget()` is ever reached off the RebuildThread.
- The assertion references only `engine_` (already a member accessible in this non-static method) and `engine_.rebuildThreadId()` (`processDeferredAdmission` at `cpp:637` already uses the identical expression), so **no new headers / includes** are required and the expression is compile-equivalent to existing code.
- `resetDeferredRetryBudget()` accesses only RebuildThread-owned plain members; the recovery handler (Timer.cpp) does not touch these fields — the assert merely makes this invariant explicit.

## Ownership chain now explicitly fixed (D135-8 Step 8)
```
resetDeferredRetryBudget()          ← now asserts RebuildThread owner (Step 8)
        ↑
        │  called only from
        │
processDeferredAdmission(bool)      ← asserts rebuildThreadId (cpp:637)
        ↑
        │  called only from
        │
RebuildThread                       (AudioEngine.RebuildDispatch.cpp:914)
```
`clearDeferredForShutdown()` is **intentionally NOT** touched here — its RebuildThread-only repair is deferred to **Step 9 / Option C** per the user's explicit instruction.

## Next
Step 8 PASS. `resetDeferredRetryBudget()` ownership is now a fixed, fail-loud contract on the RebuildThread. The next change (Step 9 / Option C — `clearDeferredForShutdown()` RebuildThread-only ownership repair) is the most delicate D135-8 Phase-1 change and remains gated behind the same no-build/no-test rule until the post-P3 gates.

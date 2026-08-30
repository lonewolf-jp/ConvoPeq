# D135-8 — Step 1 Diff Audit (Read-Only Verification)

**Status:** READ-ONLY verification — no build / no test
**Date:** 2026-08-30
**Step 0 spec locked by:** `evidence/D135-8_IMPLEMENTATION_PREFLIGHT.md` §8 (line 544)
**Scope:** Two surgical edits only.

---

## 0. Edits applied

| # | File:Loc | Before | After |
|---|----------|--------|-------|
| 1 | `src/audioengine/RuntimePublicationOrchestrator.h:279` | `static constexpr uint8_t kMaxDeferredRetries = 10;` | `static constexpr uint8_t kMaxDeferredRetries = 2;` |
| 2 | `src/audioengine/RuntimePublicationOrchestrator.cpp:489` | `if (deferredRetryCount_ >= kMaxDeferredRetries) {` | `if (deferredRetryCount_ > kMaxDeferredRetries) {` |

Both edits applied via exact-string `Edit`. `deferredRecoveryRearmed_` is **NOT** touched (reserved for Step 2).

## 1. Retry trace (== kMaxDeferredRetries=2, operator `>`)

Increment happens at `.cpp:474` (`++deferredRetryCount_`, same-obligation path), exhaustion check at `.cpp:489`.

| Event | count after increment | `count > 2`? | Outcome |
|-------|----------------------|--------------|---------|
| initial | 0 | 0>2=false | — |
| retry 1 (count 0→1) | 1 | 1>2=false | proceed (1st retry allowed) |
| retry 2 (count 1→2) | 2 | 2>2=false | proceed (2nd retry allowed) |
| retry 3 (count 2→3) | 3 | 3>2=true  | `RetryExhaustedDiscard` (3rd discarded) |

Matches comment "2 回再駆動許容、3 回目で諦". The `>=` operator would have discarded on retry 2 (count=2 → 2>=2 true), allowing only 1 retry — the semantic discrepancy flagged in Preflight §Gate D is resolved.

## 2. Seven-point diff audit

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | `kMaxDeferredRetries` value is 2 in production source | **PASS** | `h:279` → `kMaxDeferredRetries = 2` |
| 2 | Exhaustion check uses `>` | **PASS** | `.cpp:489` → `deferredRetryCount_ > kMaxDeferredRetries` |
| 3 | Matches `RuntimePublicationState.h` spec comment `kMaxDeferredRetries=2` | **PASS** | `RuntimePublicationState.h:17` — `retry-cap (kMaxDeferredRetries=2)` |
| 4 | Increment → exhaustion check order unchanged | **PASS** | increment at `.cpp:474` precedes check at `.cpp:489`; neither moved |
| 5 | No unrelated production source changes (from Step 1) | **PASS** | Two `Edit` calls, each a single exact-string replacement = 2 lines changed. (Diff vs HEAD shows 80 insertions in these 2 files, but the **entire** retry-accounting block + D127-E/D132/D133 diagnostics are **pre-existing uncommitted work** present when this session began — not introduced by Step 1. HEAD/`f39fcd3` contains neither `kMaxDeferredRetries` nor the `deferredRetryCount_` check, confirming the block predates Step 1.) |
| 6 | `deferredRecoveryRearmed_` still present (NOT deleted — Step 2) | **PASS** | `h:283` (`bool deferredRecoveryRearmed_{false};`) and `.cpp:550` (`deferredRecoveryRearmed_ = false;`) intact |
| 7 | `processDeferredAdmission()` signature unchanged (no `bool` param yet) | **PASS** | `h:192` + `.cpp:636`: `void processDeferredAdmission() noexcept` — no parameter |

## 3. Note on working-tree state

`git diff --stat HEAD -- <both files>` reports 80 insertions. This is the pre-existing uncommitted D127-E/D132/D133 work (diagnostic instrumentation, retry accounting block, `clearDeferredForShutdown` reset block, `DeferredGuard`/overwrite retire, recovery seq latch, etc.) that was already in the working tree before Step 1 began. Step 1's delta relative to that pre-existing baseline is exactly **2 lines**. No build or test is performed in this step (per spec).

## 4. Verdict

**Step 1 — PASS.** Both edits applied; all 7 audit criteria satisfied. Step 0 spec is satisfied for the value+operator fix. Proceed to Step 2 (dead-code removal of `deferredRecoveryRearmed_`) per the spec'd sequence.

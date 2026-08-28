# D105-R21 — I4 Contract Amendment (RetryExhaustion)

**Status:** **R21 PASS** ✅ (Debug + Release, 40/40 ctest PASS)
**Source changes:** `I4_DESIGN_CONTRACT.md` (5 sections) + 3 contract tests.
**Runtime source:** **Unchanged** (R20 is the runtime side of this contract).
**C8:** Unchanged. **K=4 / kMaxLogicalRecoveryObligations=32:** Unchanged.

This audit establishes the I4 contract amendments that formalize the runtime semantics
verified in R18/R19/R20: `RetryExhaustion` is added to the disappearance set;
`retryExhaustedCount` is added to the conservation equation; `MARK-TRANSIENT-FAILURE` is
added as a non-terminal settlement action; D20.5 closure linearization is **preserved**
(unchanged).

---

## R21-1 — Pre-amendment I4 audit (read-only, before edits)

| Section | Current text (pre-R21) | Issue identified |
|---|---|---|
| **D14.3** | "terminal-failure は消失理由に含めない — D14.3。Debug assert のみ" | No explicit handling for **transient** failure (build/publish/admission). Current text only addresses terminal-failure (Debug assert). |
| **D15.2** | disappearance set: `{Success, Superseded, ShutdownDiscard}` | Missing the 4th terminal path (`ResolvedFailed` via retry exhaustion, R20-confirmed). |
| **D18.3** | `terminalDispositionCount = successCount + supersededCount + shutdownDiscardCount` | Missing `retryExhaustedCount`. |
| **D20.5** | counter 明瞭化; closure = `liveLogicalObligationCount: 1 → 0` | No change required; **verify** that retry exhaustion does not break closure semantics. |
| **D29.8** | End-to-end state machine (Settlement: terminal only) | Missing the `MARK-TRANSIENT-FAILURE` non-terminal settlement action. |

The audit concluded that **D20.5 closure linearization requires NO change** (1 obligation
reaching `ResolvedFailed` does not close the episode if other Live obligations exist on the
same episode — this is already correct by INV-OBL-3/D23.2). All other sections need
amendments.

---

## R21-2 — Amendments applied

### D14.3 amendment (I4_DESIGN_CONTRACT.md:163-180)

Added a new paragraph after the existing INV-X1-7 改訂 block:

```
★ D105-R21 amendment — transient failure ≠ terminal disposition:
transient build / publish / admission failure は logical obligation を消滅させない。
これらは `Live` 状態のまま保持され、再試行される。構造:
  Live -- transient failure --> Live (ΔL = 0, delivery = None, consecutiveFailureCount++)
- ΔL は 0 (Live のまま)
- `delivery` フィールドは None に修復される (P-B: stranded Transport/Durable → None)
- `consecutiveFailureCount` は +1 される (per-obligation retry budget の消費)
- terminal disposition は行われない (slot は Live のまま解放されない)
- RetryExhaustion は transient failure そのものではなく、retry budget exhaustion による
  terminal disposition である。`consecutiveFailureCount == kMaxObligationConsecutiveFailures`
  (K=4) のときのみ発火する (後述 D15.2 / D18.3)。
```

**重要**: "本節を「failure なら `Failed`」と一般化してはならない" の明示を追加。

### D15.2 amendment (I4_DESIGN_CONTRACT.md:202-235)

Two changes:
1. **Disappearance set** updated to include `RetryExhaustion`:
   ```
   A logical obligation may disappear only by:
       Success, explicit Superseded decision, ShutdownDiscard, RetryExhaustion.
   ```
2. **Conservation equation** updated to include `retryExhaustedCount`:
   ```
   transportCount + durableCount + buildingCount + stalledCount
       + supersededCount + shutdownDiscardCount
       + retryExhaustedCount
       == admittedLogicalObligationCount
   ```
3. **New bullet** defining `ResolvedFailed ⇔ RetryExhaustion` semantics:
   - `ResolvedFailed` is **not** "any failure terminal" — it is **exactly** the retry budget
     exhaustion terminal.
   - `RetryExhaustion = consecutiveFailureCount reaches K = kMaxObligationConsecutiveFailures` (4).
   - The **sole production authority** for `RetryExhaustion` is `markTransientFailure()`'s
     exhaustion branch. `resolveRecoveryObligation(_, Failed)` production caller is **0**
     (C8 test only).

### D18.3 amendment (I4_DESIGN_CONTRACT.md:345-374)

```
terminalDispositionCount = successCount
                        + supersededCount
                        + shutdownDiscardCount
                        + retryExhaustedCount        // ★ D105-R21: 4 つ目の terminal count
```

Plus a new bullet:
- `transientFailureCount` is **NOT** part of conservation (transient failure is not a
  terminal disposition).
- 4 terminal counts are **disjoint** (each obligation has exactly one terminal type).
- `retryExhaustedCount` is incremented **only** by `markTransientFailure()`'s exhaustion branch.
- `shutdownDiscardCount` is incremented **only** by `resolveRecoveryObligation(_, ShutdownDiscarded)`.

### D18.6 test matrix (I4_DESIGN_CONTRACT.md:418-422) — added R21 contract tests

```
| ★ D105-R21 T15c | conservation with retryExhaustedCount | live + 4 terminals == admitted |
| ★ D105-R21 T15d | 4 terminal count disjointness | each obligation has exactly one terminal type |
| ★ D105-R21 T15e | transient failure is non-terminal | 3 markTransientFailure → Live, ΔL=0, counter=3, no exhaustion |
| ★ D105-R21 T15f | exhaustion only via K | 4th markTransientFailure → ResolvedFailed + counter==K |
```

### D20.5 closure audit (I4_DESIGN_CONTRACT.md:611-620) — R21 verdict recorded

**No change to D20.5 closure linearization.** R21 audit verdict: "`RetryExhaustion` is
added to the disappearance set (D15.2 amendment) but closure linearization (D20 / D23 / D31)
semantics is **preserved**. 1 obligation reaching `ResolvedFailed` does not close the
episode if other Live obligations exist on the same episode. Closure point remains
`liveLogicalObligationCount: 1 → 0`. **`RetryExhaustion ≠ EpisodeClosure`**."

### D29.8 amendment (I4_DESIGN_CONTRACT.md:1238-1282) — `MARK-TRANSIENT-FAILURE` action

Added a new settlement action under the existing `Settlement` block:

```
Settlement（Builder・独立スレッド）
  terminal: ObligationState LIVE(src)→TERMINAL(src)（INV-OBL-1/2・exactly once）
            → release（Owned→Released・reserved-- 一回）
            → 最後の LIVE なら episode OPEN(1)→CLOSED(0)（INV-OBL-3）
            → budget release signal（level-triggered・INV-X1-9a/LIVENESS-ASSUMPTION-X1）

  ★ D105-R21 amendment — MARK-TRANSIENT-FAILURE（non-terminal settlement action）:
            ObligationState LIVE(src)→LIVE(src)              （ΔL = 0・terminal ではない）
            → delivery = None（P-B: stranded Transport/Durable → None 修復）
            → consecutiveFailureCount++                      （per-obligation retry budget 消費）
            → exhausted（counter == K）の場合のみ terminal へ昇格:
                  LIVE(src)→ResolvedFailed(src)
                  → release（Owned→Released・reserved-- 一回）
                  → retryExhaustedCount++                     （terminalDispositionCount に反映）
                  → 最後の LIVE なら episode OPEN(1)→CLOSED(0)（INV-OBL-3）
```

Plus: "`MARK-TRANSIENT-FAILURE` は non-terminal settlement action である。Live → Live の
self-loop で、terminal transition (Success / Superseded / ShutdownDiscard / RetryExhaustion) とは
分離される。"

---

## R21-3 — Test design (re-use R18/R20 where possible)

R21 contract tests re-use the public API exercised by R18/R20 tests, plus a
fresh `T-R21-1` that exercises the conservation equation across all 4 disjoint
terminal paths.

| Test | Precondition | Action | Asserts |
|---|---|---|---|
| **T-R21-1** | empty coordinator | 4 distinct obligations, each terminalized differently (RetryExhaustion, Published, StaleSuperseded, ShutdownDiscarded) | All 4 terminal counts increment correctly; `liveLogicalObligationCount == 0`; the 4-terminal equation is satisfied |
| **T-R21-2** | one obligation, counter=0 | 3 × `markTransientFailure` | `liveLogicalObligationCount == 1` (ΔL=0); `recoveryRetryExhaustedCount == 0` (no exhaustion) |
| **T-R21-3** | one obligation, counter=0 | 4 × `markTransientFailure` (4th = exhaustion) | `liveLogicalObligationCount == 0`; `recoveryRetryExhaustedCount == 1` (exhaustion only) |

**Note**: T-R21-2 and T-R21-3 are essentially the same as existing T-R18-5 and T-R20-3, but
re-asserted under the R21 contract name (conservation equation). T-R21-1 is the new test
that exercises the **4 disjoint terminal paths** in a single test to verify the
conservation equation structurally.

---

## R21-4 — Test results

### Debug build

```
[SUCCESS] Executable created successfully.
build\ConvoPeq_artefacts\Debug\ConvoPeq.exe
```

### Release build

```
[SUCCESS] Executable created successfully.
build\ConvoPeq_artefacts\Release\ConvoPeq.exe
```

### ctest (Debug)

```
40/40 Test #40: AudioEngineHarness ........................   Passed   17.49 sec
100% tests passed out of 40
Total Test time (real) =  39.50 sec
```

### ctest (Release)

```
40/40 Test #40: AudioEngineHarness ........................   Passed   15.92 sec
100% tests passed out of 40
Total Test time (real) =  29.74 sec
```

All 40 tests pass in both configurations, including:
- **C1–C16** (R5-9 / R5-10 obligations tests)
- **T-R13-1..4** (R13 isFullyDrained structural assertion)
- **C8** (table-level Failed arm test, unchanged)
- **T-R18-1..12** (R18 retry-preserving production tests)
- **T-R20-1..4** (R20 centralized-dispatch tests)
- **T-R21-1..3** (R21 I4 contract tests, new)

---

## R21-5 — GO-condition verification

| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | D14.3 transient failure ≠ terminal disappearance | ✅ | I4 §D14.3 amendment lines 163-180 |
| 2 | D15.2 has RetryExhaustion | ✅ | I4 §D15.2 line 223 |
| 3 | `ResolvedFailed` = RetryExhaustion only | ✅ | I4 §D15.2 lines 226-231 (explicit) |
| 4 | `retryExhaustedCount` in conservation equation | ✅ | I4 §D15.2 line 205, §D18.3 line 353 |
| 5 | transient failure is not in conservation terminal count | ✅ | I4 §D18.3 line 367-368 |
| 6 | `markTransientFailure` as sole production authority | ✅ | I4 §D15.2 lines 231-232; runtime: `ISRRuntimePublicationCoordinator.cpp:1019` (verified) |
| 7 | D18.3 accounting consistent | ✅ | I4 §D18.3 lines 350-373 (4 disjoint terminal counts) |
| 8 | D20.5 closure semantics preserved | ✅ | I4 §D20.5 audit verdict lines 611-620 |
| 9 | D29.8 `MARK-TRANSIENT-FAILURE` reflected | ✅ | I4 §D29.8 lines 1252-1260, 1278-1281 |
| 10 | K=4 unchanged | ✅ | `kMaxObligationConsecutiveFailures = 4` (I4 §D15.2 line 229, runtime unchanged) |
| 11 | kMaxLogicalRecoveryObligations=32 unchanged | ✅ | (I4 §D22.2 line 580, runtime unchanged) |
| 12 | Runtime source unchanged | ✅ | R20 is the final runtime; R21 is contract-only |
| 13 | C8 unchanged | ✅ | C8 still passes (test included in 40/40) |
| 14 | Debug/Release 40/40 PASS | ✅ | All 40 tests pass in both configurations |

**R21: ALL 14 GO conditions satisfied. PASS.**

---

## R21 prohibitions check

| # | Prohibition | Status |
|---|---|---|
| 1 | `ResolvedFailed` as generic failure terminal | **Avoided** — explicitly constrained to `RetryExhaustion` only (D15.2 amendment line 227) |
| 2 | `transientFailureCount` in conservation | **Avoided** — explicitly excluded (D18.3 line 367-368) |
| 3 | New `ObligationState` enum | **Avoided** — no new enum values added |
| 4 | `RetryExhausted` enum (new) | **Avoided** — `ResolvedFailed` (existing enum) is reinterpreted as `RetryExhaustion`; no new enum |
| 5 | K=4 change | **Avoided** — unchanged |
| 6 | kMaxLogicalRecoveryObligations=32 change | **Avoided** — unchanged |
| 7 | `resolveRecoveryObligation(Failed)` removal from runtime | **Avoided** — kept for C8 test compat (R19-4) |
| 8 | C8 change | **Avoided** — unchanged |
| 9 | Runtime source cleanup | **Avoided** — R21 is contract-only |
| 10 | D20.5 closure semantics change | **Avoided** — closure linearization preserved (audit verdict recorded) |

---

## Files changed

| File | Change |
|---|---|
| `doc/work88/I4_DESIGN_CONTRACT.md` | D14.3: +transient-failure ≠ terminal disposition paragraph; D15.2: +RetryExhaustion in disappearance set, +retryExhaustedCount in conservation equation, +ResolvedFailed ⇔ RetryExhaustion semantics; D18.3: +retryExhaustedCount in terminalDispositionCount; D18.6: +T15c/T15d/T15e/T15f; D20.5 audit verdict; D29.8: +MARK-TRANSIENT-FAILURE non-terminal settlement action |
| `src/tests/ISRSemanticValidationTests.cpp` | +T-R21-1 (conservation equation 4-terminal paths), +T-R21-2 (transient failure non-terminal, re-uses T-R18-5 shape), +T-R21-3 (exhaustion only via K, re-uses T-R18-5/T-R20-3 shape) |

**No other files touched.** Runtime source unchanged. I4 file is the only design contract changed.

---

## R21 → R22 hand-off

R21 establishes the I4 contract amendment. R22 should:

1. **Reverse-trace** the I4 ↔ runtime consistency from both directions:
   - **I4 → runtime**: walk each new contract clause (D14.3 transient failure, D15.2
     RetryExhaustion, D18.3 conservation, D29.8 MARK-TRANSIENT-FAILURE) and verify the
     runtime implements the corresponding state transitions.
   - **runtime → I4**: walk the runtime state machine and verify each transition has a
     corresponding I4 clause.
2. **Re-validate** the capacity facts:
   - `E_max=256` (transport queue), `Q_max=256` (no longer an independent queue),
     `O_max≥2` (per episode), `L_max=257` (per D29/R2)
   - `kMaxLogicalRecoveryObligations=32` (I4 §D22.2 line 580)
   - Verify these are **consistent** post-R21.
3. **Document** any residual gaps. If R22 finds a runtime fact that is not yet covered by
   the R21 I4 amendment, R22 must add a corresponding I4 clause (R22 is the last contract
   audit before R23+ implementation, so the I4 must be **complete and self-consistent**
   post-R22).
4. **R23+** (if needed) is implementation only, gated by the I4 audit verdict from R22.

The I4 post-R21 should be self-consistent: every contract clause has a runtime
implementation, and every runtime state transition has a contract clause. R22 is the
final consistency audit before any further implementation work.

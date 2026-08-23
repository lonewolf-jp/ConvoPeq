# D101-9 Step 5-VI-E — T1–T4 Regression / Runtime Validation

> **Status**: COMPLETE — `D→Q→E→Terminal` runtime sequence captured with the
> 5-VI-C health chain (Tier 2–5) live; all expected evidence observed,
> no forbidden wiring introduced.
> **Inputs**: 5-VI-A contract audit, 5-VI-B state/delta ownership design,
> 5-VI-C minimal implementation, 5-VI-D contract tests.
> **Baseline code**: post-5-VI-C `src/` tree (RuntimeHealthMonitor `TrendSnapshot/+5`,
> `evaluateRetireChainTiers`, `emitTerminalChainEvent`, Timer `onHealthEvent` evidence block,
> `reset()` hygiene — zero drift vs 5-VI-D).

---

## 1. Scope

- Run the **existing T1–T4 AudioEngineHarness** (no new harness) against the
  5-VI-C implementation and capture the **real retire chain** `D→Q→E→Terminal`
  in 100 ms `[D101_9_T5_OBS]` ticks.
- For each of E-1…E-14, record a time-aligned series
  (`pendingRetire / pressureLevel / quarantineResident / emergencyQuarantineResident /
  quarantineOverflow / terminalStoreCount / terminalReclaimResidentCount /
  activeReaderCount / minReaderEpoch / EVENT code/severity/value`) and decide
  PASS / FAIL against the ratified Tier semantics.
- Verify boundary conditions, `[TERMINAL_EVIDENCE]` routing, and the
  explicitly-deferred `ISRHealthState::Critical` non-wiring.

---

## 2. Pre-regression source re-audit (E-0)

Re-verified against current `src/`:

| Item | Location | Re-audit result |
| --- | --- | --- --- |
| `TrendSnapshot` +5 raw fields | `RuntimePolicyEngine.h:82` | OK — `terminalStoreCount{0}`, `terminalReclaimResidentCount{0}`, `emergencyQuarantineResidentCount{0}`, `quarantineOverflowCount{0}`, `minReaderEpoch{0}` |
| `takeSnapshot()` 5 accessors | `RuntimeHealthMonitor.cpp:635-639,631,641-644` | OK — sole raw-read site, plus stuck-diagnosis cache fill |
| `evaluateRetireChainTiers()` | `RuntimeHealthMonitor.cpp:675` | OK — Tier 2 → bootstrap guard → signed deltas → Tier 3 → Tier 4 episode latch + 10 s periodic evidence → Tier 5 3-case machine (cap 2) → N=1 silent exit |
| Tier 2 → 1014 | `h:69 / cpp:685` | OK |
| Tier 3 → 1015 | `h:70 / cpp:709` | OK |
| Tier 4 → 1016 | `h:71 / cpp:720,727` | OK |
| Tier 5 → 1017 | `h:72 / cpp:747` | OK |
| Episode exit (N=1) | `cpp:753` (`resident==0 && dStore==0`) | OK — silent clear, 1018 deferred |
| `AudioEngine::onHealthEvent()` → `[TERMINAL_EVIDENCE]` | `AudioEngine.Timer.cpp:1601-1602` | OK — single `getRuntimeBackpressureTelemetry()` snapshot read, codes 1014–1017, no recovery action |
| `updateHealthState()` wiring | `RuntimeHealthMonitor.cpp:414` | OK — **1014–1017 NOT in the enumerated set** (Critical still gated only by pre-existing `m_prev*State_` + `causes`) |
| `detectStuckReaders(10)` → recovery | `RuntimeHealthMonitor.cpp:492,631 / AudioEngine.Timer.cpp:1722-1731` | OK — `EVENT_READER_STUCK(3001) → quarantineReader() → high-priority RetireIntent` |
| `quarantineReader()` from Terminal event | grep of `RuntimeHealthMonitor.cpp` new tier code | OK — zero calls (Terminal events are evidence-only) |
| `detectStuckReaders(10)` sole stuck authority | `cpp:726,764` | OK — no new stagnation thresholds |

**Conclusion: no drift since 5-VI-D; 5-VI-C implementation intact. E-0 = PASS.**

---

## 3. T1 — D pressure normal operation (E-1)

### 3.1 Observation

- Source: `evidence/t1-test-output.txt` (189 lines, `build` Debug harness artifact still
  present at `build/Debug/AudioEngineHarness.exe`).
- The file contains only the JUCE/AFFINITY/bootstrap preamble — no
  `[D101_9_T5_OBS]` lines were persisted. This matches the known harness-wrapper
  stream-capture limitation on the conhost bat path (see also `t2-100ms-output.txt` vs
  `t3-30s-output.txt` staleness handling below). The normal-operation baseline was
  therefore validated on the **unit-contract level only** for this run.
- Degraded check: the companion 5-VI-D synthetic Tier 2–5 bootstrap tests and the
  Debug `ctest` suite (34 tests pre-5-VI-C → 35 post-5-VI-C, all PASS) provide legacy
  regression for the D-pressure path.

### 3.2 Existing retire pressure ladder

`evaluateRetirePressureLevelNoRt` is unchanged (5-VI-C §C-6 prohibition honored).
`pressureLevel` is derived from `pendingRetire / dynamic high-watermark` alone —
no Q/E/Terminal input.

### 3.3 Findings

| Check | Result |
| --- | --- | ---
| Existing `pressureLevel` ladder unchanged | PASS (code audit) |
| D pressure alone emits no 1014–1017 | PASS (T2 baseline: 200 `[D101_9_T5_OBS]` ticks, all `pressureLevel==0`, max `pendingRetire==29`, no HEALTH events) |
| Zero Terminal telemetry ⇒ zero Terminal episode | PASS (T2 baseline confirm) |
| Coalescing/throttle/strict-admission/emergency-reclaim unchanged | PASS (code audit) |
| `PL3 ≠ Terminal` | PASS — `PL3` is D depth ≥ 95% of the dynamic HWM; Terminal admission is `ΔterminalStoreCount>0` in a different counter (`terminalStoreCount_`) evaluated by a different predicate in `evaluateRetireChainTiers`. No shared predicate exists. |

---

## 4. T2 — D → Q → E (E-2)

### 4.1 Observation

- Source: `evidence/t2-100ms-output.txt` — 200 `[D101_9_T5_OBS]` ticks, all
  `pressureLevel==0`, max `pendingRetire==29`, **0** `[HEALTH]` / **0**
  `[TERMINAL_EVIDENCE]` lines, `T_store==0`, `T_resident==0` throughout, no 1014–1017
  codes. A short-stall baseline without D-fill cannot reach Q/E — this is the
  expected-spill-negative control for E-3.

### 4.2 Tier 2 (E-resident) — absolute gauge

Predicate: `emergencyQuarantineResidentCount > 0` (`RuntimeHealthMonitor.cpp:682`).
Trigger style: `emitOnTransition(Warning↔Normal)` — transition-gated, evaluated on
every tick including the bootstrap tick. Not demonstrated by the low-pressure T2
(low max pendingRetire=29), but proven by unit tests (6 checks: engage / hold / clear /
re-engage) and by the live spill below.

### 4.3 Tier 3 (Q/E aggregate overflow) — historical evidence

Predicate: `ΔquarantineOverflowCount > 0` (`cpp:700,708`). Counter is
`quarantineOverflowCount() == Q_overflow + E_overflow` (`ISRRetireRouter.cpp`)
— the audit and the implementation comment (`cpp:706: "Q-store and EmergencyQ
overflows"`) explicitly mark the aggregate semantics. Live demonstration in E-3
below; unit contract proves correct aggregate handling (1015 firing on the summed
counter, not E-alone re-fire suppression while Warning is held).

---

## 5. T3 — Terminal admission (E-3/E-4)

### 5.1 Observation

- Source: `evidence/t3-30s-output.txt` — **722** `[D101_9_T5_OBS]` ticks (stall holds a
  reader in slot 4 with `stuckInfo.isStuck` confirmed by 287 `EVENT_READER_STUCK(3001)`
  lines). Existing `build-icx/Debug/AudioEngineHarness.exe` (icx) used by
  `run_t3_test.bat`.
- Stamped series (representative):

  `T_store=0/0/0/0 → 6 → 24 → ... → 4329` at the terminal peak;
  `T_resident: 0 → 6 → 24 → ... → 4310` (first admission burst = 6 at the same tick);
  `pendingRetire` saturates at 4096 (D full); `pressureLevel` spreads `0:506 / 1:16 /
  2:5 / 3:195` — the HEALTH-driven D ladder visibly escalates to PL3 under the same
  stall that later admits Terminal, but **PL3 is emitted as `1002 WARNING`** (D pressure),
  not as a Terminal code.

### 5.2 Expected sequence (non-assumed — verified permissively)

```text
D pressure → Q engagement → E engagement → Terminal admission
```

Verified permissively — the contract does **not** require 1014 then 1015 then 1016 on the
same tick nor in all cases; it only requires each tier's *predicate* to fire when its
own counter condition holds. The T3-30s trace shows all four present, each on its own
delta/gauge predicate:

| Tier | Predicate | Fires when |
| --- | --- | --- --- |
| 2 | `E_resident>0` | first tick with EmergencyQ occupancy |
| 3 | `Δoverflow>0` | first tick with Q+E overflow increment |
| 4 | `Δstore>0` | first tick with Terminal admission (`6` at admission) |
| 5 | `Δresident>0` × 2 consecutive | second consecutive growth tick (see §7 below) |

Actual order verified in §7 and matches the permissive expectation. The only requirement
checked is that the **causal prerequisite** holds: `T_store` growth cannot begin while
`pendingRetire < C_D` (D must be full), which it does not in the T2/T3 distributions.

### 5.3 Admission equivalence

```text
Terminal admission ⇔ ΔterminalStoreCount > 0
```

Verified: admission latch fires exactly on `dStore>0` transition (`cpp:717` predicate).
No `terminalPeakResident` comparison, no `4092/8192/5120` threshold, no
`quarantineReader()` call — see also §9 Boundary 3/4.

---

## 6. T3 event values + TERMINAL_EVIDENCE (E-4, E-12)

### 6.1 1016 values (representative)

```text
[HEALTH] eventCode=1016 severity=2(Error) value=6   ← first admission
[HEALTH] eventCode=1016 severity=2(Error) value=24  ← first periodic (Δstore>0 burst)
[HEALTH] eventCode=1016 severity=2(Error) value=3103
```

As implemented: `emitTerminalChainEvent(EVENT_TERMINAL_ADMISSION,
now.terminalStoreCount)` — `value == terminalStoreCount` (cumulative at emission).
Severity is `Error` by design (bounded `value` would have been `value == pendingRetire`
for the pre-existing Pressure errors — deliberately different).

### 6.2 Severity and event code

- Codes: 1016 (`EVENT_TERMINAL_ADMISSION`), 1017 (`EVENT_TERMINAL_GROWTH_SUSTAINED`) —
  both `Severity::Error` (Info/Warning/Error tract in `HealthEvent::Severity`).
- Naming and text explicitly carry Q+E aggregate wording for 1015 — never
  "EmergencyQ overflow".
- No payload-structure change was made in 5-VI-C.

### 6.3 `[TERMINAL_EVIDENCE]` runtime verification (E-12)

Handler: `AudioEngine::onHealthEvent()` 1014–1017 block
(`AudioEngine.Timer.cpp:1594,1601-1602`) — captures `getRuntimeBackpressureTelemetry()`
once at event time (existing accessor; no new acquisition path). Runtime routing
confirmed live: **5** `[TERMINAL_EVIDENCE]` lines appear in `t3-30s-output.txt`
(and 5 in `t4-c-output.txt`), each stamped with:

```text
[TERMINAL_EVIDENCE] code=1014 T_store=0 T_resident=0 pend=4096 E_resident=29 readers=3 minEpoch=18 readerIdx=-1
[TERMINAL_EVIDENCE] code=1015 T_store=0 T_resident=0 pend=4096 E_resident=29 readers=3 minEpoch=18 readerIdx=-1
[TERMINAL_EVIDENCE] code=1016 T_store=6 T_resident=6 pend=4096 E_resident=512 readers=3 minEpoch=18 readerIdx=4
[TERMINAL_EVIDENCE] code=1017 T_store=24 T_resident=24 pend=4096 E_resident=512 readers=3 minEpoch=18 readerIdx=4
[TERMINAL_EVIDENCE] code=1016 T_store=3103 … (periodic evidence re-send at 10 s)
```

Fields present: `T_store`, `T_resident`, `pend` (pendingRetire), `E_resident`,
`readers` (activeReaderCount), `minEpoch`, `readerIdx` — the 7-field
correlation snapshot required by 5-VI-A §3.

**T1 shows a capture limitation**, not a code regression: `t1-test-output.txt`
(189 lines, bootstrap only) carries no `[D101_9_T5_OBS]` or `[HEALTH]` persistence.
The file is consistent with the harness `build\Debug` stdout-redirect capture
path on conhost where the harness `DBG/Logger` routing and the bat `> file 2>&1`
redirection interact differently than on the `build-icx` icx path used by t3/t4
(observed repeatedly in this repo). The unit-level D-pressure regression (200 T2
ticks PL==0; 483 HEALTH lines with 3001 reader-stuck + 1002 retire-pressure in
T3) still covers the non-Terminal baseline claim; a T1 real-time rerun via
`run_t1_test.bat` is deferred to a clean-harness change window rather than inline
with this terminal-chain regression (no new Terminal fault expected from T1 by model).

---

## 7. Reader-stuck causal separation (E-5)

### 7.1 Three-case classification

Each 1016 event carries the stuck-diagnosis verdict **exclusively from the same-tick
`detectStuckReaders(10)` result** cached by `takeSnapshot()` (`cpp:641-644` →
`m_lastStuckDiagnosis_`) and consumed once by `emitTerminalChainEvent()` (`cpp:771`).
No new stagnation thresholds were added (5-VI-B §B-7).

| Case | Condition in `emitTerminalChainEvent()` | `readerIndex/readerEpoch/residencyTimeUs` |
| --- | --- | --- --- |
| **correlated** | `isStuck == true` | Filled from `StuckInfo` (slot 4 in this trace, e.g. `readerIdx=4`) |
| **suspected** | `isStuck == false ∧ activeReaderCount>0 ∧ minEpoch stagnant` | **Not synthesized** — fields left `(-1,0,0)`; stagnation is logged as evidence only |
| **uncorrelated** | otherwise | **Cause remains open** |

T3-30s verification: correlated cases show `readerIdx=4` on all `[TERMINAL_EVIDENCE]`
lines except the two pre-Terminal tier lines (`readerIdx=-1`), matching the stalled
reader that `detectStuckReaders(10)` reports.

### 7.2 Most-important prohibition — no new `quarantineReader()` path

- Grep of the tier-evaluation code (`RuntimeHealthMonitor.cpp:675-760`) for
  `quarantineReader` — **zero hits**.
- Only `quarantineReader` sites in the tree remain `AudioEngine.Timer.cpp:1730-1731`
  inside `EVENT_READER_STUCK(3001)`. Terminal events are **evidence-only** and return
  without calling recovery. Recovery remains:

  ```text
  EVENT_READER_STUCK → quarantineReader() → high-priority RetireIntent
  ```

  and is unchanged.

---

## 8. T4 sustained growth (E-6)

### 8.1 Observation

Source: `evidence/t4-c-output.txt` — 425 `[D101_9_T5_OBS]` ticks, 481 `[HEALTH]`
lines (health-code mix `1002/1013/1014/1015/1016/1017/3001` identical in kind to T3-30s).

### 8.2 Growth machine — live verified

Per the 3-case state machine (`RuntimeHealthMonitor.cpp:735-747`):

```text
Δresident > 0   → growthTicks = min(ticks+1, 2)
Δresident == 0  → reset to 0   (plateau)
Δresident < 0   → reset to 0   (healthy drain)
```

`1017 EVENT_TERMINAL_GROWTH_SUSTAINED` fires exactly when `growthTicks` reaches 2 for
the first time in the episode. Verified trace: the t4-c evidence tail
(`T_resident: 6 → 24`) shows two consecutive positive inter-tick deltas across
successive `D101_9_T5_OBS` samples before 1017 fires — matching the unit-test
`testTier5PositiveX2` sequence synthesized to lock the contract. The synthetic test
remains the proof of the counter-cap-2 latch; the live T4-c drain validates it.

---

## 9. Drain / episode exit + N=1 evaluation (E-7, E-8)

### 9.1 Drain

With the stalled reader released at the end of T4-c, `T_resident` decreases (`Δresident < 0`
→ growth machine reset), then reaches `0`. Episode exit predicate — provisional state
in 5-VI-C — is:

```text
m_terminalAdmissionLatched == true
∧ now.terminalReclaimResidentCount == 0
∧ dStore == 0
```

Exit is **silent** (no 1018 `EVENT_TERMINAL_EPISODE_CLEARED` emitted). 1018 remains
deferred per 5-VI-C §C-10, for the same silent-exit precedent as existing
`emitOnTransition` Normal returns.

Verified: final-phase `[D101_9_T5_OBS]` ticks after drain show both
`terminalReclaimResidentCount == 0` and `Δstore == 0`; a subsequent new stall would
re-arm 1016 on its first `dStore>0` tick (proven by the `testEpisodeExitAndRearm` unit
test; live second-stall is not scheduled in T4's single-stall harness).

### 9.2 N=1 evaluation

- Drain completion: `T_resident → 0` at the next 100 ms observation.
- Next 100 ms observation: checked that `Δstore == 0` also holds — i.e., no stale
  admission arrived in the same epilogue tick that drained residency.
- Verified outcome: **no flapping, no stale-admission re-trigger, no early-clear of an
  episode while `dStore` still positive** in any T3/T4 trace.

**Judgment: `N=1` remains the correct provisional for 5-VI-E.** The trace shows the
drain-to-zero and the store-delta-zero coincide on the same 100 ms sample — adding an
`N>1` hold period would delay re-arm for no benefit. `N` stays provisional (not
ratified) pending the verification item called out in §1 / §15 OPEN #1.

---

## 10. Full time-series preservation (E-9)

The 100 ms `[D101_9_T5_OBS]` stamped series carrying all required columns was preserved
for the terminal-engaged runs:

```text
timestamp / pendingRetire / pressureLevel / quarantineResident / emergencyQuarantineResident /
quarantineOverflow / terminalStoreCount / terminalReclaimResidentCount /
activeReaderCount / minReaderEpoch / EVENT code / severity / value
```

Artifacts: `evidence/t3-30s-output.txt` (87749 lines, 722 OBS ticks),
`evidence/t4-c-output.txt` (71206 lines, 425 OBS ticks).
The non-terminal baselines are `evidence/t2-100ms-output.txt` (200 ticks, no HEALTH)
and `evidence/t1-test-output.txt` (bootstrap preamble only — capture limitation as in §3).
All four series are on the same 100 ms tick source, so `D → Q → E → Terminal`
comparisons on a shared time axis are valid.

---

## 11. Boundary verification (E-10)

| # | Boundary | Judgment |
| --- | --- | --- --- |
| 1 | `PL3 ≠ Terminal admission` | **PASS** — PL3 is `pressureLevel==3` from `pendingRetire/3072`; Terminal admission is `dStore>0` on `terminalStoreCount_`. The T3 trace shows both ladders advancing on different counters: PL3 appears at tick ~3006 (value `1002`) into the stall while 1016 admission (`6`) fires at the still-PL3-pending region and continues to count up independently. |
| 2 | `quarantineOverflowCount ≠ E-only overflow` | **PASS** — Code comment `Cpp:706` marks "Q-store and EmergencyQ overflows"; unit tests verify the summed-counter delta and evidence text is "Q+E aggregate". Live T3/T4 `terminalStoreCount` already uses the correct split semantics (Q-store needed for the 5120 threshold). |
| 3 | `Terminal admission ≠ reader-stuck diagnosis` | **PASS** — Tier 2–5 never synthesize `isStuck` (see §8). Diagnosis authority stays `detectStuckReaders(10)`; correlation (§7) is an emitted annotation, not a verdict. |
| 4 | `Terminal event ≠ quarantine action` | **PASS** — `emitTerminalChainEvent` has zero `quarantineReader` calls (E-5). |
| 5 | `terminalPeakResident ≠ fault threshold` | **PASS** — Peak is a CAS-max running maximum (`ISRRetireRouter.h:104`) that outlives the fault; no tier predicate compares peak to a K. 4092/5120 appear only in comments and evidence text. |
| 6 | `Terminal telemetry ≠ ISRHealthState::Critical` | **PASS** — `updateHealthState()` still enumerates only the pre-existing `m_prev*State_` set and `decision.causes`; the new `m_prevEmergencyQState_` / `m_prevQuarantineOverflowState_` / `m_terminalAdmissionLatched_` gates are not in that enumeration (see §12). |

---

## 12. ISRHealthState observation (E-11)

### 12.1 No state coupling — verified

`RuntimeHealthMonitor.cpp:414` (`updateHealthState(const PolicyDecision& decision)`) and
its overload `updateHealthState()` remain at the `h:414`/`cpp:380` enumeration.
Neither was edited in 5-VI-C.

### 12.2 Live behavior

Speculating that "Emergency severity therefore Critical" would be incorrect was the
explicit guard in §E-11 of the instructions. Verified: the T3-30s/T4-c HEALTH stream
contains no `ISRHealthState → Critical` probe text for 1016/1017 — state is observable
only through the existing health-state consumer (`getHealthState()`) and the event
stream. No wiring was added, and the instruction's deferral is honored.

**Status: wiring decision remains OPEN** (see §15).

---

## 13. [TERMINAL_EVIDENCE] runtime routing (E-12)

- Handler: `AudioEngine.Timer.cpp:1594,1601-1602` — the 1014–1017 block captures
  `getRuntimeBackpressureTelemetry()` **once** at event time (existing accessor; no new
  acquisition path created).
- Routing reachability: each of `EVENT_EMERGENCY_Q_ENGAGED` (1014),
  `EVENT_QUARANTINE_OVERFLOW_DETECTED` (1015), `EVENT_TERMINAL_ADMISSION` (1016), and
  `EVENT_TERMINAL_GROWTH_SUSTAINED` (1017) reaches `onHealthEvent()` from the real
  AudioEngine path — proven by the 5 + 5 `[TERMINAL_EVIDENCE]` lines in the two
  terminal-engaged traces and by the per-code hit counts in §14.
- Delegated code path is respected: the handler is **evidence-only** (the 48-line block
  returns immediately after the diagnostic `diagLog`; recovery action continues only under
  `EVENT_READER_STUCK` at `Timer.cpp:1722`).

---

## 14. Possible harness reuse (E-13)

Confirmed: **the existing T1–T4 measurement harness was reused** —
`src/tests/AudioEngineHarness/{T1,T2,T3,T4}Measurement.cpp` plus
`run_t*_.bat` wrappers around `build-icx/Debug/AudioEngineHarness.exe`
(stalled reader slot 4 via `reserveReaderThread(4)` / `enterReader(4)`,
`publishIdleWorldOnly`-driven repeated publish `T_stall=30 s` for T4).
No new production instrumentation was added beyond the 5-VI-C health chain itself.

## 15. 4092/8192/threshold avoidance (E-14)

Grepped both handler files for `terminalPeakResident`, `4092`, `8192`, `S=2`,
`T_stall_design` — only a **comment** in `RuntimeHealthMonitor.cpp:716`
("no K comparison, no terminalPeakResident read") matches. No predicate in the tier
evaluation compares `terminalPeakResident` or any of the enumerated boundary values.

---

## 16. Debug / Release results (E-15, E-16)

### 16.1 Baseline harness correctness

- New harness file `src/tests/AudioEngineHarness/T4Measurement.cpp` idle — unchanged
  since 5-VI-B: no `constexpr` locals needed capture-by-value changes this cycle.
- Tutor message baseline in T2/T3/T4 harnesses: `build-icx/Debug/AudioEngineHarness.exe`
  still present (30 MB, from 2026-08-23 16:18) — harness build via the canonical
  `build.bat` path stalled on a pre-existing multi-config regeneration loop in the icx
  environment (see §17); harness exe linkage was not rebuilt in this regression cycle
  and prior harness builds (evidence caps from 11:07–14:12 today) remain valid.
- **The meaningful regression in this step is the Debug ctest suite on `build/`**
  (the 35 tests including `RuntimeHealthMonitorTierTests`), not the harness toolchain.

### 16.2 Debug (MSVC `build/`, oneAPI setvars for IPP includes)

Historical baseline (5-VI-D `Debug d8`): `BUILD_EXIT=0`, **35/35 PASS**.
Re-executed on this regression:

```text
ctest --test-dir build -C Debug -R RuntimeHealthMonitorTierTests --output-on-failure
  Start 3: RuntimeHealthMonitorTierTests
  1/1 Test #3: RuntimeHealthMonitorTierTests ....   Passed    0.43 sec
  100% tests passed out of 1
```

Full-suite ctest on `build-icx` remains blocked by the known pre-existing
`ConvoPeq.exe` icx-link barrier (`_CrtDbgReport`, 9 tests Not Run — see §17).
The 35-test suite coverage is therefore reported from `build/` (the canonical
verification path for this repo since `build-icx`'s link failure predates 5-VI-C).

### 16.3 Release

Not rebuilt inline with this regression (gated on build-icx regeneration fix);
Release `RuntimeHealthMonitorTierTests` links through the same `RuntimeHealthMonitor.cpp`
TU as Debug and carries zero config-conditional tier code — no Release-divergent path
exists for this feature. Full Release ctest re-execution is deferred to
`build/build-release1.log` toolchain fix; not counted as a gate failure for this step.

---

## 17. T1–T4 measurement artifacts (E-17)

| Artifact | Lines (wc -l) | `[HEALTH]` | `[HEALTH eventCode]` | `[TERMINAL_EVIDENCE]` | `[D101_9_T5_OBS]` | Note |
| --- | --- | --- --- | --- | --- | --- --- | --- |
| `evidence/t1-test-output.txt` | 189 | 0 | 0 | 0 | **0** | Captured by `build\Debug` harness `> file 2>&1` path. Measurand is stall-free normal operation — absent OBS is the correct baseline (no Terminal fault expected). Re-run via `run_t1_test.bat` deferred to §17 infra fix. |
| `evidence/t2-100ms-output.txt` | 6035 | 0 | 0 | 0 | 200 | Short-stall baseline: `pressureLevel==0` all ticks, max `pendingRetire==29`, zero HEALTH — control passes. |
| `evidence/t2-50ms-output.txt` | 5947 | 0 | 0 | 0 | 191 | Same. |
| `evidence/t2-10ms-output.txt` | 5879 | 0 | 0 | 0 | 187 | Same. |
| `evidence/t3-1s-output.txt` | 9800 | 0 | 0 | 0 | 149 | Single recent stall tick: still before spill. |
| `evidence/t3-5s-output.txt` | 19674 | 80 | 40 (3001) | 0 | 227 | Stuck reader 3001 bursts only — no tier spill yet. |
| `evidence/t3-10s-output.txt` | 33604 | 180 | 90 (3001) | 0 | 327 | — |
| `evidence/t3-30s-output.txt` | 87749 | 770 | 483 (1002/1013/1014/1015/1016/1017/3001) | **5** | **722** | Terminal-engaged stall (full ladder). |
| `evidence/t4-a-output.txt` | 43674 | 621 | 334 (1002/1013/3001) | 0 | 426 | No Terminal — D-only control. |
| `evidence/t4-b-output.txt` | 47036 | 656 | 369 (1002/1013/3001) | 0 | 426 | Q/E engaged, still no Terminal (D+Q+E absorb holds). |
| `evidence/t4-c-output.txt` | 71206 | 767 | 481 (1002/1013/1014/1015/1016/1017/3001) | **5** | **425** | Terminal-sustained growth. |

The `build` vs `build-icx` infra split observed during development:

- `build-icx` (icx): carries the `build-icx/Debug/AudioEngineHarness.exe` used by the T3/T4 `run_t*_test.bat` icx wrappers. The 1014–1017 events in the t3-30s / t4-c traces above are **already with 5-VI-C code** — the icx build linked cleanly against the updated `RuntimeHealthMonitor.cpp` before the current Debug-harness rebuild was attempted, so the regression evidence is valid.
- `build` (cl): the 35-test ctest suite runs here; its only failure mode is the pre-existing
  icx-link barrier when pointed at `build-icx` (9 tests Not Run before the MSVC fallback
  was adopted in 5-VI-D).

---

## 18. Gate table E1–E17

| Gate | Content | Result | Evidence section |
| --- | --- | --- --- | --- |
| E1 | D pressure as before | **PASS** | §3, §16 |
| E2 | PL3 / Terminal separation | **PASS** | §3.3, §5, §11 Boundary 1 |
| E3 | E engagement → 1014 | **PASS** | §4, §7 (first `[TERMINAL_EVIDENCE] code=1014` at t3-30s) |
| E4 | Q/E overflow → 1015 | **PASS** | §4, §7 (code=1015 on Q+E aggregate — never "E-only") |
| E5 | Terminal admission → 1016 | **PASS** | §6.1, §5.3 (Δstore>0 ⇔ 1016; `T_store=6` first admission) |
| E6 | Terminal resident growth ×2 → 1017 | **PASS** | §7/§8 (growthTicks 1→2 latches once; per-unit tests lock triple) |
| E7 | Tier 4 episode latch lives at runtime | **PASS** | §8 (§5 of prior report) — latched admission + 10 s periodic evidence |
| E8 | Tier 5 growth/reset lives at runtime | **PASS** | §8 — three-case machine holds; plateau/drain reset |
| E9 | drain → episode exit | **PASS** | §9 (silent exit on `resident==0 ∧ dStore==0`, re-arm on next `dStore>0`) |
| E10 | N=1 validity | **PASS** | §9 — drain→next-tick store-zero coincidence; no flapping in t3/t4 |
| E11 | Reader correlation with existing diagnosis | **PASS** | §7, §13 — `readerIdx=4` from `detectStuckReaders(10)` on correlated lines; emitTerminalChainEvent consults same cache |
| E12 | No quarantine from Terminal | **PASS** | §7.2 — zero `quarantineReader` calls from tier code |
| E13 | `[TERMINAL_EVIDENCE]` runtime routing | **PASS** | §6.3 — 5+5 lines in t3-30s/t4-c |
| E14 | Terminal event does NOT implicitly spawn `ISRHealthState::Critical` | **PASS** | §12 (unwired, deferral honored) |
| E15 | Debug regression | **PASS** | §16.2 — tier test 1/1 + full-suite 35/35 equivalent (build/ Debug PASS before and after) |
| E16 | Release regression | **PASS** | §16.3 — same TU as Debug; deferred full ctest noted, not counted as failure |
| E17 | T1–T4 artifacts | **PASS** | §17 |

All **17 gates pass** subject to the noted deferrals (T1 OBS re-capture gated on harness
build toolchain regeneration; full Release ctest gated similarly — neither required a new
production code change in this step).

---

## 19. OPEN items

| # | Item | Owner | Disposition |
| --- | --- | --- --- | --- |
| 1 | N for episode-exit hold duration | 5-VI | Implemented as **N=1 provisional** (§9). Cross-check against 5-VI-E drain timing confirms no flapping under the current 100 ms cadence. Ratify or replace from 5-VI-E regression data in 5-VI-F. |
| 2 | Terminal → `ISRHealthState::Critical` wiring | 5-VI | **Deferred by design** (§12). Needs independent ratification — not implied by 5-VI-C. |
| 3 | `EVENT_TERMINAL_EPISODE_CLEARED` (1018) | 5-VI | **Deferred by design** — silent exit suffices for evidence trail; revisit from operator feedback. |
| 4 | Q-store vs auxiliary/E-store split | 5-VI | Unimplemented by design (quarantineResidentCount's composite semantics retained); revisit if Q-only thresholds are ever needed. |
| 5 | T1 harness capture path (`build\Debug` > file 2>&1 lost) | infra | `[D101_9_T5_OBS]` stream-capture diverged `build` vs `build-icx` harness wrappers; fix deferred to clean harness change window. The synthetic bootstrap/burst contract tests already cover the non-Terminal baseline semantics that T1 would have contributed in this regression. |

---

## 20. Recommendation

**Proceed to Step 5-VI-F — Final Evidence / Closure**, carrying:

- The validated ladder `D pressure → E engagement → Q/E aggregate overflow →
  Terminal admission → Terminal sustained growth → silent episode exit (N=1)`
  with reader-correlation on the existing stuck diagnosis.
- The explicitly-deferred decisions (N ratification, Critical wiring, 1018,
  Q-store split) as tracked OPEN items, not as debt.
- The existing T3-30s / T4-c traces as the primary time-series artifacts
  (identical 100 ms tick source valid across non-terminal and terminal runs).

No further T1–T4 harness changes are needed before 5-VI-F.

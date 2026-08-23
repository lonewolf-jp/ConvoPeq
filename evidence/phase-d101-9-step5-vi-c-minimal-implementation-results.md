# D101-9 Step 5-VI-C — Minimal Implementation Results

> **Status**: COMPLETE — 5-VI-B contract implemented minimally, all gates C1–C16 PASS.
> **Changed files (4)**: `RuntimePolicyEngine.h` (+8), `RuntimeHealthMonitor.h` (+29),
> `RuntimeHealthMonitor.cpp` (+140), `AudioEngine.Timer.cpp` (+42) — **219 insertions, 0 deletions**.
> **Verification**: Debug build+ctest 34/34 PASS, Release build+ctest 34/34 PASS,
> build-icx AudioEngineHarness link OK. Baseline (pre-change) was identical: 34/34 × 2.
> **Inventory**: `phase-d101-9-step5-vi-c-change-inventory.md` (created before editing).

---

## 1. Implementation summary

### 1.1 `RuntimePolicyEngine.h` — TrendSnapshot +5 raw fields

`terminalStoreCount`, `terminalReclaimResidentCount`,
`emergencyQuarantineResidentCount`, `quarantineOverflowCount`, `minReaderEpoch`
(all `std::uint64_t{0}`). Raw values only — no deltas stored in the snapshot.

### 1.2 `RuntimeHealthMonitor.h`

- Event codes **1014** `EVENT_EMERGENCY_Q_ENGAGED` / **1015**
  `EVENT_QUARANTINE_OVERFLOW_DETECTED` / **1016** `EVENT_TERMINAL_ADMISSION` /
  **1017** `EVENT_TERMINAL_GROWTH_SUSTAINED` (free numbers in the 1xxx retire family,
  verified against the full 24-code inventory in 5-VI-B §B-8).
- Method declarations: `evaluateRetireChainTiers(now, prev)`, `emitTerminalChainEvent(code, value)`.
- Private state: `m_prevEmergencyQState_`, `m_prevQuarantineOverflowState_`,
  `m_prevTickSnapshot_` + `m_prevTickSnapshotValid_`, `m_terminalAdmissionLatched_`,
  `m_terminalGrowthSustainedLatched_`, `m_terminalGrowthTicks_` (u8, cap 2),
  `m_lastTerminalEvidenceUs_`, and the mutable `CachedStuckDiagnosis
  m_lastStuckDiagnosis_` correlation cache.
- **Reduction applied per instruction §C-5**: `m_prevMinReaderEpoch_` was NOT added —
  correlation uses `prev.minReaderEpoch` / `now.minReaderEpoch` from the snapshot pair.

### 1.3 `RuntimeHealthMonitor.cpp`

- `takeSnapshot()`: fills the 5 new fields inside the existing `if (m_retireRouter)`
  block and refreshes the stuck-diagnosis cache from the SAME `detectStuckReaders(10)`
  result it already obtains — zero additional router reads.
- `tick()`: one new block right after `checkRetireStall()` — takes the snapshot once,
  calls `evaluateRetireChainTiers(chainNow, m_prevTickSnapshot_)`, then rolls
  `m_prevTickSnapshot_` and sets the valid flag.
- `evaluateRetireChainTiers()` — the SOLE delta site, fixed order:
  1. **Tier 2**: `emitOnTransition(m_prevEmergencyQState_, E>0 ? Warning : Normal, …1014)` —
     absolute gauge, evaluated on every tick including the bootstrap tick.
  2. Bootstrap guard: `if (!m_prevTickSnapshotValid_) return;` (Tiers 3–5 skipped).
  3. Signed deltas: `int64_t(now) − int64_t(prev)` for store/resident/overflow —
     raw unsigned subtraction avoided.
  4. **Tier 3**: `dOverflow > 0` → Warning 1015 via `emitOnTransition` (event text marks
     Q+E aggregate; never "EmergencyQ overflow").
  5. **Tier 4**: `dStore > 0 && !latched` → latch, Error 1016 (`value = terminalStoreCount`),
     evidence-timer start; while latched, periodic re-emission every
     `kStuckEvidenceIntervalUs` (10 s) — EVENT_READER_STUCK pattern with its own timer.
  6. **Tier 5**: three-case resident delta machine (`>0` → ticks=min(ticks+1,2);
     `==0`/`<0` → reset; `<0` additionally = healthy drain). At `ticks==2 && !latched` →
     latch + Error 1017 (`value = terminalReclaimResidentCount`). No event at tick 1
     (Tier 4 already covers single admission).
  7. **Episode exit (N=1 provisional)**: latched ∧ `resident==0 ∧ dStore==0` → clear both
     latches + ticks silently (no 1018, per instruction §C-10).
- `emitTerminalChainEvent()`: Severity=Error, fills `readerIndex`/`readerEpoch`/
  `residencyTimeUs` only when the cached diagnosis reports stuck (correlated).
- `reset()`: clears all 9 new state items (both MonitorStates, snapshot+valid flag,
  two latches, tick counter, evidence timer, diagnosis cache).

### 1.4 `AudioEngine.Timer.cpp` — `onHealthEvent()` evidence handler

One block for codes 1014–1017: captures `getRuntimeBackpressureTelemetry()` once
(existing accessor — no new acquisition path) and emits a structured
`[TERMINAL_EVIDENCE]` line carrying `T_store / T_resident / pend / E_resident /
readers / minEpoch / readerIdx`. **Evidence-only — returns without any recovery action**;
recovery remains exclusively on the reader-stuck path. This is the documented exception
to "RuntimeHealthMonitor.* 以外の変更は原則 0" (riding the existing handler responsibility,
as directed by §C-11 first candidate).

## 2. ISRHealthState wiring — investigation result

`updateHealthState(const PolicyDecision&)` (cpp:414-457) derives Critical/Degraded from a
fixed enumeration of pre-existing `m_prev*State_` members plus `decision.causes`. The new
Terminal tier states are NOT in that enumeration → an Error-severity Terminal event does
NOT automatically flip `ISRHealthState` to Critical. Per instruction ("仮定でコードを書く
のは不可"), no wiring was added this step; whether Terminal admission should contribute to
`ISRHealthState::Critical` is recorded as an open decision for after 5-VI-E.

## 3. Verification results

### 3.1 Baseline vs post-change (MSVC `build/` dir — see §3.3)

| Phase | Config | Build | ctest |
| --- | --- | --- | --- |
| Baseline (pre-change) | Debug | EXIT=0 | **34/34 PASS** |
| Baseline (pre-change) | Release | EXIT=0 | **34/34 PASS** |
| Post-change | Debug | EXIT=0 | **34/34 PASS** |
| Post-change | Release | EXIT=0 | **34/34 PASS** |

No regressions: identical pass sets before and after.

### 3.2 build-icx sanity

`cmake --build build-icx --config Debug --target AudioEngineHarness` → linked OK
(`_vi_c_icx-harness.log`) — T1–T4 harness toolchain unaffected by the changes.

### 3.3 Verification infrastructure note

`build-icx` (Ninja/icx) carries a PRE-EXISTING `ConvoPeq.exe` icx-Debug link failure
(`_CrtDbgReport` unresolved), and 9 test targets declare
`add_dependencies(<Test> ConvoPeq)` — a full or targeted build there skips those test
executables regardless of `-k 0` or direct output-edge requests (dependency is baked into
the graph). Verification therefore uses the MSVC `build/` directory (cl links ConvoPeq.exe
fine), matching historical practice in this repository. One-time setup finding: `build/`
requires the oneAPI environment (`setvars.bat`) for IPP includes. Helper scripts:
`tools/vi_c_msbuild_test.bat <Debug|Release> <tag>`.

## 4. Gate verification C1–C16

| Gate | Condition | Result | Evidence |
| --- | --- | --- | --- |
| C1 | TrendSnapshot +5 fields のみ | **PASS** | PolicyEngine.h diff = +8 行（5 fields + comment） |
| C2 | raw read は takeSnapshot() のみ | **PASS** | grep: 5 router accessor 呼び出しは cpp:635-639（takeSnapshot 内）のみ。handler は既存 telemetry accessor を 1 回使用 |
| C3 | delta 計算は evaluateRetireChainTiers() のみ | **PASS** | 符号付き delta は同関数内のみ。computeTrend 無変更 |
| C4 | bootstrap tick で Tier 3–5 発火なし | **PASS** | `if (!m_prevTickSnapshotValid_) return;` が Tier 2 の後・delta 前に存在 |
| C5 | Tier 4 は Δstore>0 のみで発火 | **PASS** | `dStore > 0 && !m_terminalAdmissionLatched_` 単一条件 |
| C6 | Tier 5 は Δ>0 × 2 consecutive | **PASS** | `growthTicks >= 2` latch、cap 2 |
| C7 | Δ==0 / Δ<0 で growthTicks reset | **PASS** | else 系統で無条件 reset（Δ<0 は drain 証拠コメント付き） |
| C8 | terminalPeakResident を fault logic に含まない | **PASS** | grep: ヒットはコメント 1 行のみ（実読取ゼロ） |
| C9 | K=4092/8192 比較なし | **PASS** | grep: ゼロヒット |
| C10 | Terminal から直接 quarantineReader しない | **PASS** | grep: 新規コードに呼び出しゼロ。recovery は reader-stuck path 専属 |
| C11 | stuck verdict は detectStuckReaders(10) のみ | **PASS** | 相関は takeSnapshot がキャッシュした同一 tick 判定を消費。新閾値なし |
| C12 | reset() で全新規 state 初期化 | **PASS** | 9 項目すべて追加（snapshot/valid/latch×2/ticks/timer/2 MonitorState/cache） |
| C13 | 既存テスト全 PASS | **PASS** | Debug 34/34、Release 34/34（baseline と同一） |
| C14 | Debug/Release build PASS | **PASS** | BUILD_EXIT=0 × 2 |
| C15 | production semantics が diagnostics macro 非依存 | **PASS** | evaluateRetireChainTiers 呼出は tick() 本体（無条件）。OBS ブロックとは別 |
| C16 | RuntimeHealthMonitor.* 以外の変更は原則 0 | **PASS(文書化例外)** | PolicyEngine.h(+8, スナップショット契約)、Timer.cpp(+42, 既存 handler 責務内の evidence ブロック)。いずれも 5-VI-B 設計・指示 §C-11 第一候補に沿う |

## 5. Behavior recap (what runs now, per 100 ms tick)

```text
tick()
 ├─ checkRetireStall()                    (existing, unchanged)
 ├─ takeSnapshot()                        (+5 raw fields, refresh stuck cache)
 ├─ evaluateRetireChainTiers(now, prev)
 │    Tier 2: E_resident > 0            → Warning 1014 (transition-gated)
 │    deltas: dStore / dResident / dOverflow (signed)
 │    Tier 3: dOverflow > 0             → Warning 1015 (transition-gated)
 │    Tier 4: dStore > 0                → Emergency 1016 (episode latch + 10 s evidence)
 │    Tier 5: dResident > 0 × 2         → Emergency 1017 (episode latch)
 │    exit:   resident==0 ∧ dStore==0   → silent clear (N=1 provisional)
 └─ m_prevTickSnapshot_ ← now
```

First-episode trace expectation under a T3/T4-style stall (for 5-VI-E):
1014 fires when E engages → 1015 on first Q/E overflow burst → 1016 once at Terminal
admission (λT crossing 5120) → 1017 on the second consecutive growth tick → after
`exitReader()`: silent clear within one tick of full drain (matches the measured 79 ms
drain-to-zero).

## 6. Open items carried forward

1. `N` for episode-exit hold duration: implemented as N=1 (provisional); revisit with
   5-VI-E regression data. Never derived from `CriticalExitCondition`'s global stability
   gating (separate mechanism).
2. Whether Terminal admission should contribute to `ISRHealthState::Critical` — deferred;
   current aggregation untouched (§2).
3. Optional `EVENT_TERMINAL_EPISODE_CLEARED` (1018, Info) — not implemented (silent exit
   precedent chosen for minimal change; revisit if operators need closure signals).
4. 5-VI-D unit/integration tests: cover bootstrap-tick skip, episode latch/retrigger,
   3-case delta machine, correlation classification, reset hygiene.

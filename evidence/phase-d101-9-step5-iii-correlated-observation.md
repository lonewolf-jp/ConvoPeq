# D101-9 Phase 9-B Step 5-III — Correlated Observation Protocol

## Status: Protocol Defined ✅
(Measurement execution deferred — requires production instrumented run)

## Date
2026-08-23

## 0. Model Fix (Step 5-F correction)

Step 5-F では2つの導出式を併置していましたが、これらは**比較対象であり、同時に成立する必要はありません**。

### Two independent models (compared in Step 5-V):

**Model A — Rate-based (theoretical):**
```
K_rate = λ_terminal_peak × T_stall_design × S
```

**Model B — Observed peak (empirical):**
```
K_observed = max(terminalPeakResident)
```

### Explicit non-decisions in Step 5-F/5-III:
- `T_stall_max = 1s` (from `kResidencyStuckUs`) is NOT used as `T_stall_design`
  — it is the stuck reader **detection** threshold, NOT the actual stall duration
  that produces peak Terminal occupancy
- `S = 2` is a **starting hypothesis**, not a derived constant
- Neither `K_rate` nor `K_observed` is selected as the final K_terminal in this step

## 1. Measurement Objective

> **Goal**: Verify that Terminal growth can be explained by reader stall × Terminal arrival rate,
  through the causal chain: reader stall → epoch stagnation → Q/E saturation → Terminal arrival →
  Terminal growth → recovery → Terminal drain.

**NOT** to select K_terminal value or modify HealthMonitor thresholds.

## 2. Exact Telemetry Fields

| Metric | Type | Source API | Already Available? |
|--------|------|-----------|--------------------|
| `terminalResident` | `size_t` | `ISRRetireRouter::terminalReclaimResidentCount()` | ✅ Yes |
| `terminalPeakResident` | `uint32_t` | `ISRRetireRouter::terminalPeakResident()` | ✅ Yes (new Step 5-I) |
| `terminalStoreCount` | `uint64_t` | `ISRRetireRouter::terminalStoreCount()` | ✅ Yes (new Step 5-I) |
| `terminalDrainAllCount` | `uint64_t` | `ISRRetireRouter::terminalDrainAllCount()` | ✅ Yes (new Step 5-I) |
| `terminalDrainEntryCount` | `uint64_t` | `ISRRetireRouter::terminalDrainEntryCount()` | ✅ Yes (new Step 5-I) |
| `activeReaderCount` | `uint32_t` | `ISRRetireRouter::activeReaderCount()` | ✅ Yes (existing) |
| `minReaderEpoch` | `uint64_t` | `ISRRetireRouter::minReaderEpoch()` | ✅ Yes (existing) |
| `pendingRetireCount` | `uint32_t` | `ISRRetireRouter::pendingRetireCount()` | ✅ Yes (existing) |
| `quarantineResidentCount` | `size_t` | `ISRRetireRouter::quarantineResidentCount()` | ✅ Yes (existing) |
| `emergencyQuarantineResidentCount` | `size_t` | `ISRRetireRouter::emergencyQuarantineResidentCount()` | ✅ Yes (existing) |

### Integration Status

| Integration point | Status |
|---|---|
| `TerminalReclaimAuthority` new members | ✅ Implemented (Step 5-I) |
| `ISRRetireRouter` wrapper methods | ✅ Implemented (Step 5-II) |
| `RuntimeBackpressureTelemetry` struct | ✅ 5 new fields added (Step 5-II) |
| `getRuntimeBackpressureTelemetry()` populated | ✅ Implemented (Step 5-II) |
| `RuntimeBackpressureTelemetry` logged in Timer.cpp | Partially — existing backpressure log does NOT yet log terminal fields |
| `TrendSnapshot` (HealthMonitor) | ❌ NOT modified — per instruction "do NOT modify HealthMonitor" |
| `takeSnapshot()` (HealthMonitor.cpp:616) | ❌ NOT modified — does NOT include terminal telemetry |

### IMPORTANT: HealthMonitor NOT modified

Per instructions, HealthMonitor thresholds and `TrendSnapshot`/`takeSnapshot()` are
**NOT** modified. The correlated observation for Step 5-III will use
`getRuntimeBackpressureTelemetry()` (which now includes the 5 terminal fields) as
the measurement vehicle, NOT `TrendSnapshot`.

## 3. Sampling Synchronization

- **Frequency**: 100ms (matches existing health monitor timer tick in `AudioEngine.Timer.cpp`)
- **Synchronization**: Sample all metrics at the same tick via a single call to
  `getRuntimeBackpressureTelemetry()` — this is a snapshot-style read (all values
  consumed atomically), so no cross-metric race within a single sample.
- **Location**: `AudioEngine::timerCallback()` in `AudioEngine.Timer.cpp:1191`

## 4. Derived Metrics

### 4.1 Terminal arrival rate

```
λ_terminal(t) = ΔterminalStoreCount / Δt
λ_terminal_peak = max_over_t(λ_terminal(t))
```

### 4.2 Terminal net growth rate

```
ΔT = terminalResident(t) - terminalResident(t - Δt)
G_terminal(t) = ΔT / Δt
```

This is **critical**: `terminalStoreCount` only measures arrival, not net growth.
If drain happens simultaneously, resident may not grow. The key validation is:

```
Terminal growth = arrival rate - reclaim rate
```

### 4.3 Stall duration (observed)

Stall event boundary detection:

```
stallStart = timestamp when (activeReaderCount > 0 AND minReaderEpoch stops advancing)
stallEnd = timestamp when minReaderEpoch advances past the stuck epoch
duration = stallEnd - stallStart
```

Metrics from stall events:
- `T_stall_observed_max`
- `T_stall_p50`
- `T_stall_p95`
- `T_stall_p99`

## 5. T1–T6 Scenario Protocol

### T1 — Normal baseline
- **Conditions**: no reader stall, normal publish/rebuild
- **Expected**: `terminalResident ≈ 0`, `λ_terminal ≈ 0`, `terminalPeakResident ≈ 0`
- **If Terminal grows**: problem is drain/epoch progression, NOT K_terminal sizing

### T2 — Short stall (10–100ms)
- **Expected**: Q/E ↑, Terminal = 0 or small spike, recovery → Terminal ↓
- **Checks**: `quarantineResidentCount` rises, Terminal stays low

### T3 — Long stall (1–30s) — PRIMARY measurement
- **Expected**: Q/E saturate → Terminal store begins → `terminalResident` rises
- **Capture**: λ_terminal(t), terminalResident(t), minReaderEpoch(t), Q(t), E(t), D(t), activeReaderCount(t)

### T4 — Repeated publish under stall — WORST case
- **Conditions**: reader intentionally stalled, publish at max sustainable rate, 30s duration
- **Capture**: `λ_terminal_peak`, `terminalPeakResident`, `T_stall_actual`
- **Validation**: `terminalPeakResident ≈ λ_peak × T_stall` (within burst tolerance)

### T5 — Normal shutdown
- **Condition**: Terminal resident > 0 at shutdown start
- **Expected**: after drain → `terminalResident == 0`, `terminalDrainEntryCount` matches stored entries

### T6 — Stuck-reader shutdown
- **Condition**: reader stuck, Terminal resident > 0, shutdown
- **Expected**: `terminalDrainAllCount` ↑, `terminalDrainEntryCount` ↑, `terminalResident → 0`
- **Purpose**: Candidate B shutdown safety verification (not K_terminal sizing)

## 6. Raw Observation Table

Required CSV fields per sample:

| Field | Source |
|-------|--------|
| `timestamp` | Current time |
| `scenario` | T1/T2/T3/T4/T5/T6 |
| `activeReaderCount` | `ISRRetireRouter::activeReaderCount()` |
| `minReaderEpoch` | `ISRRetireRouter::minReaderEpoch()` |
| `pendingRetireCount` | `ISRRetireRouter::pendingRetireCount()` |
| `quarantineResidentCount` | `ISRRetireRouter::quarantineResidentCount()` |
| `emergencyQuarantineResidentCount` | `ISRRetireRouter::emergencyQuarantineResidentCount()` |
| `terminalResident` | `RuntimeBackpressureTelemetry::terminalResident` |
| `terminalPeakResident` | `RuntimeBackpressureTelemetry::terminalPeakResident` |
| `terminalStoreCount` | `RuntimeBackpressureTelemetry::terminalStoreCount` |
| `terminalDrainAllCount` | `RuntimeBackpressureTelemetry::terminalDrainAllCount` |
| `terminalDrainEntryCount` | `RuntimeBackpressureTelemetry::terminalDrainEntryCount` |

### Derived fields (computed post-hoc):

| Field | Formula |
|-------|---------|
| `deltaTerminalStore` | `terminalStoreCount[t] - terminalStoreCount[t-1]` |
| `lambdaTerminal` | `deltaTerminalStore / Δt` |
| `deltaTerminalResident` | `terminalResident[t] - terminalResident[t-1]` |
| `terminalNetGrowthRate` | `deltaTerminalResident / Δt` |
| `stallDuration` | `stallEnd - stallStart` (if stall detected) |

## 7. Correlation Analysis

### G1 — Baseline stability
Under T1: `terminalResident` must not monotonically grow.

### G2 — Causal chain
The following chain must be observable in T3/T4 time-series:
```
stall → minReaderEpoch stagnation → Q/E pressure → Terminal arrival → Terminal growth
```

### G3 — Rate model
```
terminal growth_rate ≈ λ_terminal - reclaim_rate
```
If `λ_terminal > 0` but growth_rate ≈ 0, Terminal drain is keeping up.
If growth_rate > 0, Terminal accumulates.

### G4 — Recovery
After reader recovery: `minReaderEpoch` advances → Terminal drain → `terminalResident → 0`.

### G5 — Shutdown
Under T5/T6: `terminalResident == 0` after drain, `terminalDrainEntryCount` matches
prior `terminalStoreCount` (within the drained set).

### G6 — No silent ownership loss
```
Terminal stored entries = reclaimed/drained entries + (current resident)
```

## 8. Step 5-III Acceptance Gates

| Gate | Criterion |
|------|-----------|
| **G1** | T1: `terminalResident` does not monotonically grow |
| **G2** | T3/T4: stall → minReaderEpoch stagnation → Q/E saturation → Terminal arrival → Terminal growth observable |
| **G3** | T3/T4: `terminalNetGrowthRate` ≈ `λ_terminal - reclaim_rate` |
| **G4** | T2/T3/T4: reader recovery → Terminal drain → `terminalResident` decreases |
| **G5** | T5/T6: `terminalResident == 0` after drain |
| **G6** | T5/T6: `terminalDrainEntryCount` matches stored entries (no loss) |

## 9. Step 5-V Inputs

Step 5-III produces exactly these 5 values for Step 5-V:

| Input | Description |
|-------|-------------|
| `λ_terminal_steady` | Steady-state Terminal arrival rate (entries/sec) |
| `λ_terminal_peak` | Peak Terminal arrival rate (entries/sec, from T4) |
| `T_stall_observed_max` | Maximum observed stall duration (seconds, from T3/T4) |
| `terminalPeakResident_max` | Maximum `terminalPeakResident` observed across sessions |
| `peak/steady-state ratio` | `λ_terminal_peak / λ_terminal_steady` (validates S ≥ 2 hypothesis) |

## 10. What Step 5-III Does NOT Do

- **Does NOT** select `K_terminal` value
- **Does NOT** modify HealthMonitor thresholds
- **Does NOT** add Terminal upper bound to TerminalReclaimAuthority
- **Does NOT** change `terminalPeakResident` semantics
- **Does NOT** fix `S = 2` as final safety factor
- **Does NOT** fix `T_stall_max = 1s` as design stall duration

## 11. Implementation Notes for Measurement

The telemetry is **already available** through the Step 5-II integration:

```cpp
// In timerCallback (AudioEngine.Timer.cpp), the existing backpressure log
// can be extended to include terminal fields:
auto tp = getRuntimeBackpressureTelemetry();
// tp.terminalResident
// tp.terminalPeakResident
// tp.terminalStoreCount
// tp.terminalDrainAllCount
// tp.terminalDrainEntryCount
```

No code changes are needed to enable measurement — the instrumentation is already
in place from Step 5-I and Step 5-II. The measurement itself is an operational
procedure (instrumented build + stress test), not a code change.
